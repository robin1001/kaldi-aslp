/* Audio Provider
 * Created on 2015-08-07
 * Author: zhangbinbin hechangqing
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include <iostream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "aslp-online/wav-provider.h"

#include <stdlib.h>
#include <pthread.h>

bool SendFull(int socket_id, char *data, int to_send) {
    while (to_send > 0) {
        int ret = write(socket_id, data, to_send);
        if (ret <= 0) return false;
        to_send -= ret;
        data += ret;
    }
    return true;
}

bool ReadFull(int socket_id, char* buf, int len) {
    int to_read = len;
    int has_read = 0;
    int ret;
    while (to_read > 0) {
        ret = read(socket_id, buf + has_read, to_read);
        if (ret <= 0) {
            return false;
        }
        to_read -= ret;
        has_read += ret;
    }
    return true;
}

void *GetDecodeResult(void *arg) {
    using namespace kaldi::aslp_online;
    while (true) {
        int client_sid = *static_cast<int *>(arg);
        //Get Asr Result
        int recv_len;
        char cmd;
        if (!ReadFull(client_sid, (char *)&recv_len, 4)) {
            return static_cast<void *>(NULL);
        }
        recv_len = ntohl(recv_len);
        if (!ReadFull(client_sid, &cmd, 1)) {
            return static_cast<void *>(NULL);
        }
        if (cmd == WavProvider::kPartialResult || 
                cmd == WavProvider::kFinalResult) {
            char *result = new char[recv_len];
            if (!ReadFull(client_sid, result, recv_len - 1)) {
                delete [] result;
                return static_cast<void *>(NULL);
            }
            result[recv_len-1] = 0;
            KALDI_LOG << "Asr Reslut: " << result;
            delete [] result;
        }
        if (cmd == WavProvider::kEOS) {
            break;
        }
    }
    return static_cast<void *>(NULL);
}

int main(int argc, char *argv[]) {
    using namespace kaldi;

    const char *usage = "audio-provider-client, send audio in format to audio-provider\n"
        "e.g. : ./audio-provider 127.0.0.1 10000 wav_rspecifier\n";
    ParseOptions po(usage);
    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
        po.PrintUsage();
        exit(1);
    }

    std::string server_ip = po.GetArg(1).c_str();
    int32 server_port = strtol(po.GetArg(2).c_str(), 0, 10);
    std::string wav_rspecifier = po.GetArg(3);

    //WaveData wav;
    //ifstream istream();
    //wav.Read();
    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    for (; !reader.Done(); reader.Next()) {
        int32 client_sid = socket(AF_INET, SOCK_STREAM, 0);
        if (client_sid == -1) {
            KALDI_ERR << "create client socket failed";
        }
        struct sockaddr_in server_addr;
        unsigned long addr = inet_addr(server_ip.c_str());
        //memset(server_addr, 0, sizeof(server_addr));
        server_addr.sin_addr.s_addr = addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        if (connect(client_sid, (struct sockaddr*) &server_addr, sizeof(server_addr))) {
            close(client_sid);
            KALDI_ERR << "could not connect to " << server_ip << ":" << server_port;
        }

        // Create thread to receive decode results
        pthread_t rev_tid;
        int err = pthread_create(&rev_tid, NULL, GetDecodeResult, &client_sid);
        if (err != 0)
            KALDI_ERR << "could not create thread";

        // Send audio data
        std::string wav_key = reader.Key();
        KALDI_LOG << "File: " << wav_key;
        const WaveData &wav_data = reader.Value();
        if (wav_data.SampFreq() != 16000)
            KALDI_ERR << "Sampling rates other than 16kHz are not supported!";
        int32 num_chan = wav_data.Data().NumRows();
        KALDI_ASSERT(num_chan == 1);
        const SubVector<BaseFloat> &data_vec = wav_data.Data().Row(0);
        const float *data = data_vec.Data();
        int left = data_vec.Dim();
        int packet_size = 1024;
        int data_len = 0;
        // cmd0 = 0x00 for raw wave data
        // cmd1 = 0x01 for end of data
        unsigned char cmd0 = 0x00, cmd1 = 0x01;

        while (left > 0) {
            int to_send = (left > packet_size) ? packet_size: left;
            data_len = 1 + to_send * sizeof(short);
            KALDI_VLOG(2) << "send one package, package size: " << data_len;
            data_len = htonl(data_len);
            //should use reinterpret_cast, just feel tired to type that
            SendFull(client_sid, (char *)&data_len, 4);
            SendFull(client_sid, (char *)&cmd0, 1);
            for (int i = 0; i < to_send; i++) {
                //short value = htons(static_cast<short>(*data)); 
                short value = (static_cast<short>(*data)); 
                SendFull(client_sid, (char *)&value, sizeof(short));
                data++;
            }
            left -= to_send;
            //data += to_send;
        }
        //send finish signal
        data_len = 1;
        KALDI_VLOG(2) << "send one package, package size: " << data_len;
        data_len = htonl(data_len);
        SendFull(client_sid, (char *)&data_len, 4);
        SendFull(client_sid, (char *)&cmd1, 1);

        // Wait receive thread to end
        err = pthread_join(rev_tid, NULL);
        if (err != 0)
            KALDI_ERR << "can not join with receive thread";
        //    //Get Asr final Result
        //    int recv_len;
        //    char cmd;
        //    ReadFull(client_sid, (char *)&recv_len, 4);
        //    recv_len = ntohl(recv_len);
        //    ReadFull(client_sid, &cmd, 1);
        //    char *result = new char[recv_len - 1];
        //    ReadFull(client_sid, result, recv_len - 1);
        //    KALDI_LOG << "Asr Reslut: " << result;
        //    delete [] result;
        close(client_sid);
    }
}

