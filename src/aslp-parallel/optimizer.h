/* Created on 2017-03-07
 * Author: Zhang Binbin
 */

#ifndef ASLP_PARALLEL_OPTIMIZER_H_
#define ASLP_PARALLEL_OPTIMIZER_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#

namespace kaldi {
/* 
 * Please refer "An overview of gradient descent optimization algorithms" for details
 */

class Optimizer {
public:
    Optimizer(int size): size_(size) {}
    virtual ~Optimizer() {}
    virtual void Optimize(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        KALDI_ASSERT(param != NULL);
        KALDI_ASSERT(size_ == grad.Dim()); 
        KALDI_ASSERT(size_ == param->Dim()); 
        Solve(grad, param);
    }
protected:
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) = 0;
    int size_; // tensor size
};


class SGD: public Optimizer {
public:
    SGD(int size, float lr = 0.1): Optimizer(size), lr_(lr) {}
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        param->AddVec(-lr_, grad);
    }
private:
    float lr_;
};


class Momentum: public Optimizer {
public:
    Momentum(int size, float lr = 0.1, float momentum = 0.9): 
        Optimizer(size), lr_(lr), momentum_(momentum), 
        grad_(size) {}
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        grad_.AddVec(lr_, grad, momentum_);
        param->AddVec(-1.0, grad_);
    }
private:
    float lr_, momentum_;
    CuVector<BaseFloat> grad_;
};


class AdaGrad: public Optimizer {
public:
    AdaGrad(int size, float lr = 0.01): 
        Optimizer(size), lr_(lr), 
        acc_grad_(size), grad_(size) {}
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        acc_grad_.AddVecVec(1.0, grad, grad, 1.0);
        grad_.CopyFromVec(acc_grad_);
        grad_.ApplyFloor(1e-8);
        grad_.ApplyPow(0.5);
        grad_.InvertElements();
        grad_.MulElements(grad);
        param->AddVec(-lr_, grad_);
    }
private:
    float lr_;
    CuVector<BaseFloat> acc_grad_; // Gt
    CuVector<BaseFloat> grad_;  
};


class RMSprop: public Optimizer {
public:
    RMSprop(int size, float lr = 0.001): 
        Optimizer(size), lr_(lr), 
        acc_grad_(size), grad_(size) {}
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        acc_grad_.AddVecVec(0.1, grad, grad, 0.9);
        grad_.CopyFromVec(acc_grad_);
        grad_.ApplyFloor(1e-8);
        grad_.ApplyPow(0.5);
        grad_.InvertElements();
        grad_.MulElements(grad);
        param->AddVec(-lr_, grad_);
    }
private:
    float lr_;
    CuVector<BaseFloat> acc_grad_; // Gt
    CuVector<BaseFloat> grad_;  
};


class AdaDelta: public Optimizer {
public:
    AdaDelta(int size, float gamma = 0.95): 
        Optimizer(size), gamma_(gamma), 
        acc_grad_(size), acc_delta_(size), 
        grad_(size), delta_(size) {}
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        acc_grad_.AddVecVec(1 - gamma_, grad, grad, gamma_);
        grad_.CopyFromVec(acc_grad_);
        grad_.ApplyFloor(1e-8);
        grad_.ApplyPow(0.5);
        grad_.InvertElements();
        delta_.CopyFromVec(acc_delta_);
        delta_.ApplyFloor(1e-8);
        delta_.ApplyPow(0.5);
        grad_.MulElements(delta_);
        grad_.MulElements(grad);
        param->AddVec(-1.0, grad_);
        acc_delta_.AddVecVec(1 - gamma_, grad_, grad_, gamma_);
    }
private:
    float gamma_;
    CuVector<BaseFloat> acc_grad_; // Gt
    CuVector<BaseFloat> acc_delta_; // delta t
    CuVector<BaseFloat> grad_;  
    CuVector<BaseFloat> delta_;  
};


class Adam: public Optimizer {
public:
    Adam(int size, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999): 
        Optimizer(size), lr_(lr), 
        beta1_(beta1), beta2_(beta2), 
        t_(1),
        grad_(size), delta_(size), diff_(size) {}
    virtual void Solve(const CuVector<BaseFloat> &grad, 
            CuSubVector<BaseFloat> *param) {
        grad_.AddVec(1 - beta1_, grad, beta1_);
        delta_.AddVecVec(1 - beta2_, grad, grad, beta2_);
        float s1 = 1.0 / (1 - pow(beta1_, t_));
        float s2 = 1.0 / (1 - pow(beta2_, t_));
        diff_.CopyFromVec(delta_);
        diff_.Scale(s2);
        diff_.ApplyFloor(1e-8);
        diff_.ApplyPow(0.5);
        diff_.InvertElements();
        diff_.MulElements(grad_);
        param->AddVec(-lr_ * s1, diff_);
        t_++;
    }
private:
    float lr_;
    float beta1_, beta2_;
    int t_;
    CuVector<BaseFloat> grad_; // m(t)
    CuVector<BaseFloat> delta_; // v(t)
    CuVector<BaseFloat> diff_;  
};

struct OptimizerOption {
    std::string solver;
    float lr;
    float momentum;
    float adagrad_lr, rmsprop_lr, adam_lr;
    float adadelta_gamma;
    float adam_beta1, adam_beta2;
    OptimizerOption() : solver("momentum"), lr(0.01), momentum(0.9),
        adagrad_lr(0.01), rmsprop_lr(0.001), adam_lr(0.001),
        adadelta_gamma(0.95),
        adam_beta1(0.9), adam_beta2(0.999) {}
    void Register(OptionsItf *opts) {
        opts->Register("solver", &solver,
                        "Optimizer solver(sgd | momentum | adagrad | adadelta | rmsprop | adam)");
        opts->Register("lr", &lr,
                        "learning rate for (sgd | momentum) optimizer");
        opts->Register("sgd-momentum", &momentum,
                        "momentum for (momentum) optimizer");
        opts->Register("adagrad-lr", &adagrad_lr,
                        "learning rate for (adagrad) optimizer");
        opts->Register("rmsprop-lr", &rmsprop_lr,
                        "learning rate for (rmsprop) optimizer");
        opts->Register("adam-lr", &adam_lr,
                        "learning rate for (adam) optimizer");
        opts->Register("adadelta-gamma", &adadelta_gamma, 
                        "update factor for (adadelta) optimizer");
        opts->Register("adam-beta1", &adam_beta1, 
                        "update mean factor for (adam) optimizer");
        opts->Register("adam-beta2", &adam_beta2, 
                        "update variance factor for (adam) optimizer");
    }

    Optimizer *NewInstance(int size) const {
        if (solver == "sgd") {
            return new SGD(size, lr);
        } 
        else if (solver == "momentum") {
            return new Momentum(size, lr, momentum);
        }
        else if (solver == "adagrad") {
            return new AdaGrad(size, adagrad_lr);
        }
        else if (solver == "rmsprop") {
            return new RMSprop(size, rmsprop_lr);
        }
        else if (solver == "adadelta") {
            return new AdaDelta(size, adadelta_gamma);
        }
        else if (solver == "adam") {
            return new Adam(size, adam_lr, adam_beta1, adam_beta2);
        }
        else {
            KALDI_ERR << "Unknown solver type " << solver;
            return NULL;
        }
    }
};

} // end of namespace kaldi
#endif

