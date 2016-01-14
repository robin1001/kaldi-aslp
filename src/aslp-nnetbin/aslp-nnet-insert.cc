// Copyright 2016  ASLP (Author: zhangbinbin liwenpeng duwei)

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-affine-transform.h"

namespace kaldi {
namespace aslp_nnet {

int32 IndexOfLastUpdatableComponent(const Nnet &nnet) {
  int32 index = -1, nc = nnet.NumComponents();
  for (int32 c = 0; c < nc; c++) {
    if (nnet.GetComponent(c).IsUpdatable()) {
      //if (index != -1) return -1; // >1 softmax components.
      //else index = c;
      index = c;
    }
  }
  return index;
}

void InsertComponents(const Nnet &src_nnet,
                      int32 c_to_insert, // component-index before which to insert.
                      Nnet *dest_nnet) {
  KALDI_ASSERT(c_to_insert >= 0 && c_to_insert <= dest_nnet->NumComponents());
  int32 c_tot = dest_nnet->NumComponents() + src_nnet.NumComponents();
  std::vector<Component*> components(c_tot);
  for (int32 c = 0; c < c_to_insert; c++)
    components[c] = dest_nnet->GetComponent(c).Copy();
  for (int32 c = 0; c < src_nnet.NumComponents(); c++)
    components[c + c_to_insert] = src_nnet.GetComponent(c).Copy();
  for (int32 c = c_to_insert; c < dest_nnet->NumComponents(); c++)
    components[c + src_nnet.NumComponents()] = dest_nnet->GetComponent(c).Copy();
  // Re-initialize "dest_nnet" from the resulting list of components.

  // The Init method will take ownership of the pointers in the vector:
  dest_nnet->Destroy();
  for (int32 c = 0; c < components.size(); c++) {
      dest_nnet->AppendComponent(components[c]);
  }
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;

    const char *usage =
        "Insert components into a neural network-based acoustic model.\n"
        "Usage:  aslp-nnet-insert [options] <model-in1> <...> <model-inN> <model-out>\n"
        "e.g.:\n"
        " aslp-nnet-insert 1.nnet \"aslp-nnet-init hidden_layer.config -| \" 2.nnet\n";
    
    ParseOptions po(usage);
    
    bool binary_write = true;
    bool randomize_next_component = true;
    int32 insert_at = -1;
    BaseFloat stddev_factor = 0.1;
    int32 srand_seed = 0;
    
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("randomize-next-component", &randomize_next_component,
                "If true, randomize the parameters of the next component after "
                "what we insert (which must be updatable).");
    po.Register("insert-at", &insert_at, "Inserts new components before the "
                "specified component (note: indexes are zero-based).  If <0, "
                "inserts before the last updatable component(typically before the softmax).");
    po.Register("stddev-factor", &stddev_factor, "Factor on the standard "
                "deviation when randomizing next component (only relevant if "
                "--randomize-next-component=true");
    po.Register("srand", &srand_seed, "Seed for random number generator");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        raw_nnet_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    //TransitionModel trans_model;
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      //trans_model.Read(ki.Stream(), binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    Nnet src_nnet; // the one we'll insert.
    ReadKaldiObject(raw_nnet_rxfilename, &src_nnet);

    if (insert_at == -1) {
      if ((insert_at = IndexOfLastUpdatableComponent(nnet)) == -1)
        KALDI_ERR << "We don't know where to insert the new components: "
            "the neural net doesn't have exactly one softmax component, "
            "and you didn't use the --insert-at option.";
      //insert_at--; // we want to insert before the linearity before
      // the softmax layer.
    }
    
    InsertComponents(src_nnet, insert_at, &nnet);
    KALDI_LOG << "Inserted " << src_nnet.NumComponents() << " components at "
              << "position " << insert_at;

    if (randomize_next_component) {
      int32 c = insert_at + src_nnet.NumComponents();
      Component *component = &(nnet.GetComponent(c));
      AffineTransform *uc = dynamic_cast<AffineTransform*>(component);
      if (!uc)
        KALDI_ERR << "You have --randomize-next-component=true, but the "
                  << "component to randomize is not updatable: "
                  << component->Info();
      int32 outDim, inputDim;
      outDim = uc->OutputDim();
      inputDim = uc->InputDim();
      BaseFloat stddev = stddev_factor /
          std::sqrt(static_cast<BaseFloat>(inputDim));
      //CuMatrix<BaseFloat> linear_params(uc->GetLinearity());
      //CuMatrix<BaseFloat> temp_linear_params(uc->GetLinearity());
      CuMatrix<BaseFloat> linear_params(outDim, inputDim);
      CuMatrix<BaseFloat> temp_linear_params(outDim, inputDim);
      temp_linear_params.SetRandn();
      linear_params.AddMat(stddev, temp_linear_params);
      uc->SetLinearity(linear_params);

      //CuVector<BaseFloat> bias_params(uc->GetBias());
      //CuVector<BaseFloat> temp_bias_params(uc->GetBias());
      CuVector<BaseFloat> bias_params(uc->GetBias().Dim());
      CuVector<BaseFloat> temp_bias_params(uc->GetBias().Dim());
      temp_bias_params.SetRandn();
      bias_params.AddVec(stddev, temp_bias_params);
      uc->SetBias(bias_params);
      KALDI_LOG << "Randomized component index " << c << " with stddev "
                << stddev;
    }


    {
      Output ko(nnet_wxfilename, binary_write);
      //trans_model.Write(ko.Stream(), binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Write neural-net acoustic model to " <<  nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


