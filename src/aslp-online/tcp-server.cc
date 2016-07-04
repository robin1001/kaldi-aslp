#include "aslp-online/tcp-server.h"

namespace kaldi {

TcpServer::TcpServer() {
  server_desc_ = -1;
}

bool TcpServer::Listen(int32 port) {
  h_addr_.sin_addr.s_addr = INADDR_ANY;
  h_addr_.sin_port = htons(port);
  h_addr_.sin_family = AF_INET;

  server_desc_ = socket(AF_INET, SOCK_STREAM, 0);

  if (server_desc_ == -1) {
    KALDI_ERR << "Cannot create TCP socket!";
    return false;
  }

  int32 flag = 1;
  int32 len = sizeof(int32);
  if( setsockopt(server_desc_, SOL_SOCKET, SO_REUSEADDR, &flag, len) == -1){
    KALDI_ERR << "Cannot set socket options!\n";
    return false;
  }

  if (bind(server_desc_, (struct sockaddr*) &h_addr_, sizeof(h_addr_)) == -1) {
    KALDI_ERR << "Cannot bind to port: " << port << " (is it taken?)";
    return false;
  }

  if (listen(server_desc_, 1) == -1) {
    KALDI_ERR << "Cannot listen on port!";
    return false;
  }

  std::cout << "TcpServer: Listening on port: " << port << std::endl;

  return true;

}

TcpServer::~TcpServer() {
  if (server_desc_ != -1)
    close(server_desc_);
}

int32 TcpServer::Accept() {
  std::cout << "Waiting for client..." << std::endl;

  socklen_t len = sizeof h_addr_;

  int32 client_desc = accept(server_desc_, (struct sockaddr*) &h_addr_, &len);
  if (client_desc == -1)
    KALDI_ERR << "Cannot accept";

  struct sockaddr_storage addr;
  char ipstr[20];

  len = sizeof addr;
  getpeername(client_desc, (struct sockaddr*) &addr, &len);

  struct sockaddr_in *s = (struct sockaddr_in *) &addr;
  inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);

  std::cout << "TcpServer: Accepted connection from: " << ipstr << std::endl;

  return client_desc;
}

} // namespace kaldi
