/* Wav Provider
 * Created on 2015-09-01
 * Author: zhangbinbin 
 *         hechangqing
 */
#include "aslp-online/wav-provider.h"

namespace kaldi {
namespace aslp_online {

WavProvider::WavProvider(int client_sid): client_sid_(client_sid), done_(false), connect_(true) {

}

WavProvider::~WavProvider() {
  if (client_sid_ != -1)
    close(client_sid_);
}

bool WavProvider::ReadFull(char* buf, int32 len) {
  KALDI_ASSERT(client_sid_ >= 0);
  int32 to_read = len;
  int32 has_read = 0;
  int32 ret;
  while (to_read > 0) {
    ret = read(client_sid_, buf + has_read, to_read);
    if (ret <= 0) {
      connect_ = false;
      return false;
    }
    to_read -= ret;
    has_read += ret;
  }
  return true;
}

bool WavProvider::WriteFull(const char *buf, int to_send) const{
  KALDI_ASSERT(client_sid_ >= 0);
  while (to_send > 0) {
    int ret = send(client_sid_, buf, to_send, MSG_NOSIGNAL);
    if (ret <= 0) {
      return false;
    }
    to_send -= ret;
    buf += ret;
  }
  return true;
}

bool WavProvider::Done() const {
  return (done_ && data_queue_.empty()) || (!connect_ && data_queue_.empty());
}

void WavProvider::Reset() {
  client_sid_ = -1;
  done_ = false;
  connect_ = false;
  while(!data_queue_.empty()) data_queue_.pop();
}

/* packet format 
 * len [4 bytes] + cmd [1 byte] + data[N byte]
 * cmd: 0x00 audio data + short * N
 *    : 0x01 finish signal
 */
bool WavProvider::ReadOnce() {
  int32 len;
  //Read Len first 
  if (!ReadFull((char *)&len, 4)) {
    return false;
  }
  len = ntohl(len); //convert netword to host
  KALDI_VLOG(2) << "new package arrived, package size: " << len;
  char *data = new char[len];  
  if (!ReadFull(data, len)) {
    delete [] data;
    return false;
  }
  int cur = 0;
  switch (data[0]) {
    case 0x00:{
        cur += 1;
        KALDI_ASSERT((len - 1) % sizeof(short) == 0); //2 byte
        short value;
        for (int i = 0; i < (len - 1) / sizeof(short) ; i++) {
          //value = ntohs(*(short *)(data + cur));
          value = (*(short *)(data + cur));
          data_queue_.push(value);
          cur += sizeof(short);
        }
      }
      break;
    case 0x01:
      KALDI_ASSERT(len == 1);
      done_ = true;
      break;
  }
  delete [] data;
  return true;
}

int WavProvider::ReadAudio(int num, std::vector<BaseFloat> *data) {
  data->clear();
  while (num > 0) {
    if (Done()) break; // done and no more data Or connect break and no more data
    if (!done_ && data_queue_.empty()) //no data in queue
      ReadOnce(); 
    if (!data_queue_.empty()) {
      data->push_back(static_cast<BaseFloat>(data_queue_.front()));
      data_queue_.pop();
      num--;
    }
  }
  return data->size();
}

void WavProvider::WriteDecoding() {
    int len = htonl(1);
    WriteFull((char *)&len, 4);
    char cmd = kDecoding;
    WriteFull((char *)&cmd, 1);
}

void WavProvider::WritePartialReslut(std::string result) {
    int len = htonl(1 + result.size());
    WriteFull((char *)&len, 4);
    char cmd = kPartialResult;
    WriteFull((char *)&cmd, 1);
    WriteFull(result.c_str(), result.size());
}

void WavProvider::WriteFinalReslut(std::string result) {
    int len = htonl(1 + result.size());
    WriteFull((char *)&len, 4);
    char cmd = kFinalResult;
    WriteFull((char *)&cmd, 1);
    WriteFull(result.c_str(), result.size());
}

void WavProvider::WriteEndPointing() {
    int len = htonl(1);
    WriteFull((char *)&len, 4);
    char cmd = kEndPoint;
    WriteFull((char *)&cmd, 1);
}

void WavProvider::WriteEOS() {
    int len = htonl(1);
    WriteFull((char *)&len, 4);
    char cmd = kEOS;
    WriteFull((char *)&cmd, 1);
}

void WavProvider::WritePuncResult(std::string result) {
    int len = htonl(1 + result.size());
    WriteFull((char *)&len, 4);
    char cmd = kPunctuationResult;
    WriteFull((char *)&cmd, 1);
    WriteFull(result.c_str(), result.size());
}

} // namespace aslp_online
} // namespace kaldi
