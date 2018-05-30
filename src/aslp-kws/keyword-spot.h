/*
 * Created on 2018-02-05
 * Author: Zhang Binbin
 */
#ifndef KEYWORD_SPOT_H_
#define KEYWORD_SPOT_H_

#include "utils.h"
#include "fst.h"

#include <float.h>
#include <math.h>

namespace kaldi {
namespace kws {

const int kMaxTokenPassingFrames = 100 * 60 * 10; // 10 minitue

class KeywordSpot {
public:
    KeywordSpot(const Fst &fst, const SymbolTable &filler_table):
            fst_(fst),
            filler_table_(filler_table),
            num_frames_(0),
            spot_threshold_(0.5), 
            min_keyword_frames_(0),
            min_frames_for_last_state_(5) {
        prev_tokens_.resize(fst_.NumStates(), Token());  
        cur_tokens_.resize(fst_.NumStates(), Token());  
        Reset();
    }

    void SetSpotThreshold(float threshold) {
        spot_threshold_ = threshold; 
    }

    void SetMinKeywordFrames(int frames) {
        min_keyword_frames_ = frames;
    }

    void Reset() {
        for (int i = 0; i < prev_tokens_.size(); i++) {
            prev_tokens_[i].Reset();
        }
        for (int i = 0; i < cur_tokens_.size(); i++) {
            cur_tokens_[i].Reset();
        }
        prev_tokens_[0].active = true;
        num_frames_ = 0;
    }

    // 0 garbage, 1 silence now
    bool IsFillerPhone(int phone) {
        return filler_table_.HaveId(phone);
    }

    bool Spot(const float *am_score, int num, float *confidence, 
              int32_t *keyword_id) {
        bool spot = false;
        *confidence = 0.0;
        *keyword_id = 0;

        for (int i = 0; i < prev_tokens_.size(); i++) {
            if (prev_tokens_[i].active) {
                for (const Arc *arc = fst_.ArcStart(i); arc != fst_.ArcEnd(i); arc++) {
                    CHECK(arc->next_state >= 0);
                    CHECK(arc->next_state < cur_tokens_.size());
                    CHECK(arc->ilabel <= num);
                    float score = logf(am_score[arc->ilabel - 1]); // 0 for <eps>
                    bool is_filler = IsFillerPhone(arc->ilabel);
                    bool is_self_arc = (i == arc->next_state);
                    int32_t olabel = arc->olabel;
                    cur_tokens_[arc->next_state].Update(prev_tokens_[i], olabel,
                        is_self_arc, is_filler, score);
                }
            }
        }
        
        // find best final score
        int best_state = 0, best_final_state = 0;
        float best_score = cur_tokens_[0].score, best_final_score = 0.0f;
        bool reach_final = false;
        for (int i = 1; i < cur_tokens_.size(); i++) {
            if (cur_tokens_[i].active && best_score < cur_tokens_[i].score) {
                best_score = cur_tokens_[i].score;
                best_state = i;
            }

            if (cur_tokens_[i].active && fst_.IsFinal(i)) {
                if (!reach_final) {
                    best_final_state = i;
                    best_final_score = cur_tokens_[i].score;
                    reach_final = true;
                } else if (best_final_score < cur_tokens_[i].score) {
                    best_final_state = i;
                    best_final_score = cur_tokens_[i].score;
                }
            }
        }

        // if we reach final state, then get confidence
        if (reach_final) {
            *confidence = expf(cur_tokens_[best_final_state].average_max_keyword_score);
            *keyword_id = cur_tokens_[best_final_state].keyword;
            if (cur_tokens_[best_final_state].num_keyword_frames >= min_keyword_frames_ && 
                cur_tokens_[best_final_state].num_frames_of_current_state >= min_frames_for_last_state_ &&
                *confidence > spot_threshold_) {
                spot = true;
            }
        }

        prev_tokens_.swap(cur_tokens_);
        for (int i = 0; i < cur_tokens_.size(); i++) {
            cur_tokens_[i].Reset();
        }

        num_frames_++;
        // Reset state to avoid number overflow, and it's not in a keyword state
        if (num_frames_ > kMaxTokenPassingFrames 
                && (prev_tokens_[best_state].is_filler)) {
            for (int i = 0; i < cur_tokens_.size(); i++) {
                prev_tokens_[i].Reset();
            }
        }
        return spot;
    }

    struct Token {
    public:
        Token(): active(false), 
                 is_filler(true), 
                 score(0), 
                 num_keyword_frames(0),
                 average_keyword_score(0.0), 
                 keyword(0),
                 num_frames_of_current_state(0),
                 num_keyword_states(0),
                 max_score_of_current_state(0.0),
                 average_max_keyword_score(0.0),
                 average_max_keyword_score_before(0.0) {}

        void Reset() {
            active = false;
            is_filler = true;
            score = 0;
            num_keyword_frames = 0;
            average_keyword_score = 0.0;
            keyword = 0;
            num_frames_of_current_state = 0;
            num_keyword_states = 0;
            max_score_of_current_state = 0.0;
            average_max_keyword_score = 0.0;
            average_max_keyword_score_before = 0.0;
        }

        void Update(const Token &prev, int32_t olabel, bool is_self_arc,
                    bool is_filler, float am_score) {
            // first time access by previous token
            if (!active || (active && score < prev.score + am_score)) {
                this->score = prev.score + am_score;
                // it's a keyword state
                if (!is_filler) {
                    int t = prev.num_keyword_frames;
                    average_keyword_score = (am_score + prev.average_keyword_score * t) / (t + 1); 
                    num_keyword_frames = t + 1;
                    if (is_self_arc) {
                        num_frames_of_current_state = prev.num_frames_of_current_state + 1;
                        num_keyword_states = prev.num_keyword_states;
                        max_score_of_current_state = std::max(prev.max_score_of_current_state, am_score);
                        average_max_keyword_score_before = prev.average_max_keyword_score_before;
                        CHECK(num_keyword_states > 0);
                    } else {
                        num_frames_of_current_state = 1;
                        num_keyword_states = prev.num_keyword_states + 1;
                        max_score_of_current_state = am_score;
                        average_max_keyword_score_before = prev.average_max_keyword_score;
                    }
                    average_max_keyword_score = (max_score_of_current_state + 
                                                 average_max_keyword_score_before * (num_keyword_states - 1)) / 
                                                 num_keyword_states;
                    if (olabel != 0) keyword = olabel;
                }
            }
            active = true; 
            this->is_filler = is_filler;
        }
        
        bool active;
        bool is_filler;
        float score; 

        int num_keyword_frames;
        float average_keyword_score;

        int32_t keyword;
        int num_frames_of_current_state; 

        int num_keyword_states;
        float max_score_of_current_state;
        float average_max_keyword_score;
        float average_max_keyword_score_before;
    };

private:
    const Fst &fst_; // determined fst
    // the left is filler phone/state, such as silence or <gbg> 
    const SymbolTable &filler_table_; 
    int num_frames_;
    // Make tokens the same size as number states of Fst
    std::vector<Token> prev_tokens_; 
    std::vector<Token> cur_tokens_;

    float spot_threshold_;
    int min_keyword_frames_;
    int min_frames_for_last_state_;
};

}
}

#endif
