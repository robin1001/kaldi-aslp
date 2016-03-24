// nnet/nnet-component.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
// Copyright 2016  ASLP (Author: zhangbinbin liwenpeng duwei)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "aslp-nnet/nnet-component.h"

#include "aslp-nnet/nnet-activation.h"
#include "aslp-nnet/nnet-affine-transform.h"
#include "aslp-nnet/nnet-linear-transform.h"
#include "aslp-nnet/nnet-convolutional-component.h"
#include "aslp-nnet/nnet-max-pooling-component.h"
#include "aslp-nnet/nnet-various.h"
#include "aslp-nnet/nnet-lstm-projected-streams.h"
#include "aslp-nnet/nnet-blstm-projected-streams.h"

#include "aslp-nnet/nnet-batch-normalization.h"
#include "aslp-nnet/nnet-io.h"
#include "aslp-nnet/nnet-recurrent-component.h"

#include <sstream>

namespace kaldi {
namespace aslp_nnet {

const struct Component::key_value Component::kMarkerMap[] = {
  // Actviation
  { Component::kSoftmax,"<Softmax>" },
  { Component::kBlockSoftmax,"<BlockSoftmax>" },
  { Component::kSigmoid,"<Sigmoid>" },
  { Component::kTanh,"<Tanh>" },
  { Component::kDropout,"<Dropout>" },
  // Varios
  { Component::kLengthNormComponent,"<LengthNormComponent>" },
  { Component::kSplice,"<Splice>" },
  { Component::kCopy,"<Copy>" },
  { Component::kAddShift,"<AddShift>" },
  { Component::kRescale,"<Rescale>" },
  //  
  { Component::kAffineTransform,"<AffineTransform>" },
  { Component::kLinearTransform,"<LinearTransform>" },
  { Component::kConvolutionalComponent,"<ConvolutionalComponent>"},
  { Component::kMaxPoolingComponent, "<MaxPoolingComponent>"},
  { Component::kLstmProjectedStreams,"<LstmProjectedStreams>"},
  { Component::kBLstmProjectedStreams,"<BLstmProjectedStreams>"},
  // Aslp
  { Component::kBatchNormalization, "<BatchNormalization>"},
  { Component::kInputLayer, "<InputLayer>"},
  { Component::kOutputLayer, "<OutputLayer>"},
  { Component::kScaleLayer, "<ScaleLayer>"},
  { Component::kLstm, "<Lstm>"},
  { Component::kBLstm, "<BLstm>"},
};


const char* Component::TypeToMarker(ComponentType t) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if (kMarkerMap[i].key == t) return kMarkerMap[i].value;
  }
  KALDI_ERR << "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::MarkerToType(const std::string &s) {
  std::string s_lowercase(s);
  std::transform(s.begin(), s.end(), s_lowercase.begin(), ::tolower); // lc
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    std::string m(kMarkerMap[i].value);
    std::string m_lowercase(m);
    std::transform(m.begin(), m.end(), m_lowercase.begin(), ::tolower);
    if (s_lowercase == m_lowercase) return kMarkerMap[i].key;
  }
  KALDI_ERR << "Unknown marker : '" << s << "'";
  return kUnknown;
}


Component* Component::NewComponentOfType(ComponentType comp_type,
                      int32 input_dim, int32 output_dim) {
  Component *ans = NULL;
  switch (comp_type) {
    case Component::kAffineTransform :
      ans = new AffineTransform(input_dim, output_dim); 
      break;
    case Component::kLinearTransform :
      ans = new LinearTransform(input_dim, output_dim); 
      break;
    case Component::kConvolutionalComponent :
      ans = new ConvolutionalComponent(input_dim, output_dim);
      break;
    case Component::kLstmProjectedStreams :
      ans = new LstmProjectedStreams(input_dim, output_dim);
      break;
    case Component::kBLstmProjectedStreams :
      ans = new BLstmProjectedStreams(input_dim, output_dim);
      break;
    case Component::kSoftmax :
      ans = new Softmax(input_dim, output_dim);
      break;
    case Component::kBlockSoftmax :
      ans = new BlockSoftmax(input_dim, output_dim);
      break;
    case Component::kSigmoid :
      ans = new Sigmoid(input_dim, output_dim);
      break;
    case Component::kTanh :
      ans = new Tanh(input_dim, output_dim);
      break;
    case Component::kDropout :
      ans = new Dropout(input_dim, output_dim); 
      break;
    case Component::kLengthNormComponent :
      ans = new LengthNormComponent(input_dim, output_dim); 
      break;
    case Component::kSplice :
      ans = new Splice(input_dim, output_dim);
      break;
    case Component::kCopy :
      ans = new CopyComponent(input_dim, output_dim);
      break;
    case Component::kAddShift :
      ans = new AddShift(input_dim, output_dim);
      break;
    case Component::kRescale :
      ans = new Rescale(input_dim, output_dim);
      break;
    case Component::kMaxPoolingComponent :
      ans = new MaxPoolingComponent(input_dim, output_dim);
      break;
    // Aslp extention component
    case Component::kBatchNormalization :
      ans = new BatchNormalization(input_dim, output_dim);
      break;
    case Component::kInputLayer :
      ans = new InputLayer(input_dim, output_dim);
      break;
    case Component::kOutputLayer :
      ans = new OutputLayer(input_dim, output_dim);
      break;
    case Component::kScaleLayer:
      ans = new ScaleLayer(input_dim, output_dim);
      break;
    case Component::kLstm:
      ans = new Lstm(input_dim, output_dim);
      break;
    case Component::kBLstm:
      ans = new BLstm(input_dim, output_dim);
      break;
    case Component::kUnknown :
    default :
      KALDI_ERR << "Missing type: " << TypeToMarker(comp_type);
  }
  return ans;
}


Component* Component::Init(const std::string &conf_line) {
  std::istringstream is(conf_line);
  std::string component_type_string;
  int32 input_dim, output_dim;

  // initialize component w/o internal data
  ReadToken(is, false, &component_type_string);
  ComponentType component_type = MarkerToType(component_type_string);
  ExpectToken(is, false, "<InputDim>");
  ReadBasicType(is, false, &input_dim); 
  ExpectToken(is, false, "<OutputDim>");
  ReadBasicType(is, false, &output_dim);
  Component *ans = NewComponentOfType(component_type, input_dim, output_dim);
  // Optional Id and 
  if (conf_line.find("<Id>") != std::string::npos) {
    int32 id;
    ExpectToken(is, false, "<Id>");
    ReadBasicType(is, false, &id);
    std::string input_string;
    ExpectToken(is, false, "<Input>");
    ReadToken(is, false, &input_string);
    std::vector<std::string> sub_input_string;
    SplitStringToVector(input_string, ",", true, &sub_input_string);
    int32 num_input = sub_input_string.size();
    std::vector<int32> input(num_input, 0),
                       offset(num_input, 0);
    // Parse inputs
    for (int i = 0; i < num_input; i++) {
      std::vector<std::string> field;
      SplitStringToVector(sub_input_string[i], ":", true, &field);
      KALDI_ASSERT(field.size() >= 1);
      KALDI_ASSERT(field.size() <= 2);
      ConvertStringToInteger(field[0], &input[i]);
      if (field.size() > 1) {
        ConvertStringToInteger(field[1], &offset[i]);
      }
      KALDI_VLOG(3) << "layerId " << id << " "
                    << "inputId " << input[i] << " "
                    << "offset " << offset[i];
    }
    ans->SetId(id);
    ans->SetInput(input);
    ans->SetOffset(offset);
  }

  // initialize internal data with the remaining part of config line
  ans->InitData(is);
  return ans;
}


Component* Component::Read(std::istream &is, bool binary) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF) return NULL;

  ReadToken(is, binary, &token);
  // Skip optional initial token
  if(token == "<Nnet>") {
    ReadToken(is, binary, &token); // Next token is a Component
  }
  // Finish reading when optional terminal token appears
  if(token == "</Nnet>") {
    return NULL;
  }

  ReadBasicType(is, binary, &dim_out); 
  ReadBasicType(is, binary, &dim_in);

  int32 id;
  std::vector<int32> input, offset;
  ReadBasicType(is, binary, &id);
  ReadIntegerVector(is, binary, &input);
  ReadIntegerVector(is, binary, &offset);
  KALDI_ASSERT(input.size() == offset.size());

  Component *ans = NewComponentOfType(MarkerToType(token), dim_in, dim_out);
  ans->ReadData(is, binary);
  ans->SetId(id);
  ans->SetInput(input);
  ans->SetOffset(offset);
  return ans;
}


void Component::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, Component::TypeToMarker(GetType()));
  WriteBasicType(os, binary, OutputDim());
  WriteBasicType(os, binary, InputDim());
  WriteBasicType(os, binary, id_);
  WriteIntegerVector(os, binary, input_);
  WriteIntegerVector(os, binary, offset_);

  if(!binary) os << "\n";
  this->WriteData(os, binary);
}


} // namespace aslp_nnet
} // namespace kaldi
