/* Wav Provider
 * Created on 2015-09-01
 * Last Edit  2015-09-02
 * Author: zhangbinbin 
 *         hechangqing
 */

#ifndef ASR_ASR_ONLINE_WAV_PROVIDER_H
#define ASR_ASR_ONLINE_WAV_PROVIDER_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <queue>

#include "util/common-utils.h"

namespace kaldi {

class WavProvider {
 public:
  // This class is used by decoding server.
  // This class receives raw audio and sends the recognition result.
  WavProvider(int client_sid);
  ~WavProvider();

  bool IsConnected() const { return connect_; }
  int ReadAudio(int num, std::vector<BaseFloat> *data);
  void Reset();
  // This function returns true when there is no more data.
  // The calling of function ReadAudio() will return zero after 
  // Done() returns true.
  bool Done() const;
  /* write status to client, packet format 
   * len [4 bytes] + cmd [1 byte] + data[N byte]
   * cmd: 0x00 decoding 
   *    : 0x01 partial result + result str[N byte]
   *    : 0x02 final result + result str[N byte]
   *    : 0x03 end point detected, tell client to stop sending speech data
   *    : 0x04 end of sentence, tell client to stop receving recognition result
  */
  enum { kDecoding = 0x00,
         kPartialResult  = 0x01,  
         kFinalResult = 0x02,
         kEndPoint = 0x03,
         kEOS = 0x04,
         kPunctuationResult = 0x05 };
  void WritePartialReslut(std::string result);
  void WriteFinalReslut(std::string result);
  // Add on 2016-01-25, add punctuation predict support
  void WritePuncResult(std::string); 
  void WriteEndPointing(); //detect endpoint
  void WriteDecoding();
  void WriteEOS();
 private:
  bool ReadFull(char* buf, int32 len);
  bool WriteFull(const char *buf, int to_send) const; 
  bool ReadOnce();
 private:
  int client_sid_; //client socket id
  bool done_; //if the remote audio done
  bool connect_; // whether the remote client is connected
  std::queue<short> data_queue_;
};

} // end of namespace kaldi

#endif // ASR_ASR_ONLINE_WAV_PROVIDER_H
