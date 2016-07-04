#ifndef ASLP_ONLINE_TCP_SERVER_H
#define ASLP_ONLINE_TCP_SERVER_H

#include "base/kaldi-types.h"
#include "base/kaldi-error.h"

#include <iostream>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace kaldi {
/*
 * This class is for a very simple TCP server implementation
 * in UNIX sockets.
 */
class TcpServer {
 public:
  //typedef kaldi::int32 int32;
  
  TcpServer();
  ~TcpServer();
  
  bool Listen(int32 port);  //start listening on a given port
  int32 Accept();  //accept a client and return its descriptor

 private:
  struct sockaddr_in h_addr_;
  int32 server_desc_;
};

} // namespace kaldi

#endif // ASLP_ONLINE_TCP_SERVER_H
