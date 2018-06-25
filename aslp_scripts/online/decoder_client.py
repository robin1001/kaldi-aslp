import struct
import socket
from socket import SHUT_RDWR
import threading
import json
import logging
from tornado.websocket import WebSocketClosedError
# Created on 2015-08-20
# Author: zhangbinbin hechangqing

class DecoderClient:
    def __init__(self, web_handler):
        logging.debug('new decoder client create')
        self.handler = web_handler
        self.final_results = []
        ip, port = self.handler.decoder_info
        logging.debug('DecoderClient: connecting decoder %s:%d' % (ip, port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            #set timeout to a small value to prevent from waiting too long
            #to wait for connect
            self.sock.settimeout(8.9)
            self.sock.connect((ip, port))
            #set socket to blocking mode
            self.sock.setblocking(1)
            self.thread = threading.Thread(target=self.recv_thread, args=())
            self.thread.start()
        except socket.error as err:
            logging.warning('DecoderClient: can not connect decoder %s:%d' %(ip, port))
            self.handler.handle_decoder_error_with_lock()

    def send_wav_data(self, wav_data): #0x00
        if not self.handler.decoder_error:
            data_len = 1 + len(wav_data)
            data = struct.pack('!iB', data_len, 0x00)
            assert(len(data) == 5)
            self.sock.sendall(data)
            self.sock.sendall(wav_data)

    def send_eos(self): #end of speech, cmd 0x01
        if not self.handler.decoder_error:
            data = struct.pack('!iB', 1, 0x01)
            assert(len(data) == 5)
            self.sock.sendall(data)

    def recv(self, length):
        data = []
        left = length
        while left > 0:
            tmp = self.sock.recv(left)
            if len(tmp) == 0:
                return None
            data.append(tmp)
            left -= len(tmp)
        return ''.join(data)

    def recv_thread(self):
        looping = True
        try:
            while looping:
                pack = self.recv(4 + 1)
                if pack == None: #decoder error
                    self.handler.handle_decoder_error_with_lock()
                    break
                assert(len(pack) == 5)
                data_len, cmd = struct.unpack('!iB', pack)
                data_len -= 1
                #read from decoder then parse and send to client
                if cmd == 0x00: #decoding
                    data = '{"status": 0}' 
                elif cmd == 0x01: #partial result
                    result = self.recv(data_len)
                    data = '{"status": 1, '+ '"result": "' + result + '"}'
                    logging.warning('partial result: ' + result) 
                elif cmd == 0x02: #final result
                    result = self.recv(data_len)
                    data = '{"status": 2, '+ '"result": "' + result + '"}'
                    self.final_results.append(result)
                    logging.warning('final result: ' + result) 
                elif cmd == 0x03: #endpoint
                    data = '{"status": 3}' 
                elif cmd == 0x04: #no more result
                    looping = False
                    data = '{"status": 4}' 
                elif cmd == 0x05: #punctuation result
                    result = self.recv(data_len)
                    data = '{"status": 5, '+ '"result": "' + result + '"}'
                    logging.warning('punctuation result: ' + result) 
                else: #just ignore
                    data = '{status: 6}' 
                self.handler.write_message(data)
        #web socket client error, write_message when client close the connection
        except WebSocketClosedError as ws_error:
            logging.warning('DecoderClient:recv_thread:WebSocketClosedError: ' 
                + str(ws_error))
            self.sock.shutdown(SHUT_RDWR)
        except socket.error as err:
            logging.warning('DecoderClient:recv_thread:SocketError: ' 
                + str(err))
            self.handler.handle_decoder_error_with_lock()
        finally:
            logging.debug('DecoderClient: recv_thread exit')

