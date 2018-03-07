/*
 * Created on 2018-02-05
 * Author: Zhang Binbin
 */
#ifndef KEYWORD_SPOT_H_
#define KEYWORD_SPOT_H_

#include "utils.h"
#include "fsm.h"

#include <float.h>
#include <math.h>

namespace kaldi {
namespace kws {

const int kMaxTokenPassingFrames = 100 * 60 * 10; // 10 minitue

class KeywordSpot {
public:
    KeywordSpot(const Fsm &fsm): fsm_(fsm), num_frames_(0),
            spot_threshold_(0.9), min_keyword_frames_(0) {
        prev_tokens_.resize(fsm_.NumStates(), Token());  
        cur_tokens_.resize(fsm_.NumStates(), Token());  
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
    bool IsKeywordPhones(int phone) {
        if (phone > 1) return true;
        else return false;
    }

    bool Spot(const float *am_score, int num, float *confidence = NULL) {
        bool spot = false;
        float keyword_confidence = 0.0;
        // viterbi token passing 
        for (int i = 0; i < prev_tokens_.size(); i++) {
            if (prev_tokens_[i].active) {
                for (const Arc *arc = fsm_.ArcStart(i); arc != fsm_.ArcEnd(i); arc++) {
                    CHECK(arc->next_state >= 0);
                    CHECK(arc->next_state < cur_tokens_.size());
                    CHECK(arc->ilabel < num);
                    float score = logf(am_score[arc->ilabel]);
                    bool is_keyword_phone = IsKeywordPhones(arc->ilabel);
                    cur_tokens_[arc->next_state].Update(prev_tokens_[i],
                        is_keyword_phone, score);
                }
            }
        }
        
        // find best score
        int best_state = 1;
        float best_score = cur_tokens_[1].score;
        for (int i = 2; i < cur_tokens_.size(); i++) {
            if (cur_tokens_[i].active && best_score < cur_tokens_[i].score) {
                best_score = cur_tokens_[i].score;
                best_state = i;
            }
        }
        // if we get best score at final state, then get confidence
        if (fsm_.IsFinal(best_state)) {
            keyword_confidence = expf(cur_tokens_[best_state].average_keyword_score);
            if (cur_tokens_[best_state].num_keyword_frames > min_keyword_frames_ && 
                keyword_confidence > spot_threshold_) {
                spot = true;
            }
        }
        prev_tokens_.swap(cur_tokens_);
        for (int i = 0; i < cur_tokens_.size(); i++) {
            cur_tokens_[i].Reset();
        }
        if (confidence != NULL) *confidence = keyword_confidence;

        num_frames_++;
        // Reset state to avoid number overflow, and it's not in a keyword state
        if (num_frames_ > kMaxTokenPassingFrames 
                && (!prev_tokens_[best_state].keyword_phone)) {
            for (int i = 0; i < cur_tokens_.size(); i++) {
                prev_tokens_[i].Reset();
            }
        }
        return spot;
    }

    struct Token {
    public:
        Token(): active(false), keyword_phone(false), score(0), 
                 average_keyword_score(0.0), 
                 num_keyword_frames(0) {}
        void Reset() {
            active = false;
            keyword_phone = false;
            score = 0;
            average_keyword_score = 0.0;
            num_keyword_frames = 0;
        }

        void Update(const Token &prev, bool is_keyword_phone, float am_score) {
            // first time access by previous token
            if (!active || (active && score < prev.score + am_score)) {
                score = prev.score + am_score;
                if (is_keyword_phone) {
                    int t = prev.num_keyword_frames;
                    average_keyword_score = (am_score + prev.average_keyword_score * t) / (t + 1); 
                    num_keyword_frames = t + 1;
                }
            }
            active = true; 
            keyword_phone = is_keyword_phone;
        }
        
        bool active;
        bool keyword_phone;
        float score; 
        float average_keyword_score;
        int num_keyword_frames;
    };

private:
    const Fsm &fsm_; // determined fsm
    int num_frames_;
    // Make tokens the same size as number states of Fsm
    std::vector<Token> prev_tokens_; 
    std::vector<Token> cur_tokens_;

    float spot_threshold_;
    int min_keyword_frames_;
};

}
}

#endif
