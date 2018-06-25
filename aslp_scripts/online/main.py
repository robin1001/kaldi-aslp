#!/usr/bin/python

import os
import sys
import threading
import time
import wave
import logging
import ConfigParser
import struct
import socket
import tornado.ioloop
import tornado.web
import tornado.websocket
from socket import SHUT_RDWR
from database import DataBase
from decoder_client import DecoderClient

# Created on 2015-08-19
# Author: zhangbinbin hechangqing

# TODO 
# 1. write wav big endian
# 2. all in config(wavfile path, port, log_file)
# 3. decoder error handler


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('static/demos/mob.html')

class AdminHandler(tornado.web.RequestHandler):
    def get(self):
        num_connection = self.application.num_connection
        items = []
        decoders = self.application.decoders
        for key, value in self.application.decoders_info.items():
            ip, port = key
            available = "No"
            max_thread, free_thread = value, value
            link = '/restart?ip=%s&port=%d' % (ip, port)
            if key in self.application.decoders:
                available = "Yes"
                free_thread = self.application.decoders[key]
                link = None
            items.append((ip, port, max_thread, free_thread, available, link)) 
        self.render("static/admin/admin-pretty.html", num_connection=num_connection, items=items)

class AdminConnectionsHandler(tornado.web.RequestHandler):
    def get(self):
        num_connection = self.application.num_connection
        items = []
        for connection in self.application.connections:
            ip, user_agent, t = connection
            items.append((ip, user_agent, int(time.time() - t))) 
        self.render("static/admin/admin-connections.html", num_connection=num_connection, items=items)

class AdminLogHandler(tornado.web.RequestHandler):
    def get(self):
        num_connection = self.application.num_connection
        #content = "This is log file"
        with open(self.application.log_file) as fid:
            content = fid.readlines()
        if len(content) > 15:
            content = content[len(content)-15 : -1]
        self.render("static/admin/admin-log.html", num_connection=num_connection, content=content)

class AdminHistoryHandler(tornado.web.RequestHandler):
    def get(self):
        page = int(self.get_argument('page', 0, strip=True))
        page_items = int(self.get_argument('page_items', 10, strip=True))
        num_connection = self.application.num_connection
        history = self.application.db.get_all()
        num_pages = (len(history) - 1) / page_items + 1;
        if page < 0: page = 0
        if page >= num_pages: page = num_pages - 1
        start = page * page_items 
        start_page = page - 5
        if start_page < 0: start_page = 0
        end_page = page + 5
        if end_page > num_pages: end_page = num_pages - 1
        #print start_page, page, end_page
        history = history[start: start+page_items]
        #print history
        self.render("static/admin/admin-history.html", 
                    num_connection = num_connection, 
                    history = history,
                    page_items = page_items,
                    page = page,
                    num_pages = num_pages,
                    start_page = start_page,
                    end_page = end_page)

class AdminSearchHandler(tornado.web.RequestHandler):
    def get(self):
        num_connection = self.application.num_connection
        text = self.get_argument('input', '', strip = True) 
        page = int(self.get_argument('page', 0, strip = True))
        page_items = int(self.get_argument('page_items', 10, strip = True))
        if text == '':
            result = []
            start = 0
            num_pages = 0
            start_page = end_page = 0
        else:
            # Personal sql injection
            if text[0:4] == 'sql:':
                logging.info('sql cmd >> ' + text[4:])
                try:
                    result = self.application.db.execute_sql(text[4:])
                except:
                    result = [('SQL Syntax Error!!! --->>> ', text[4:])] 
                #print result
            # Find all match the pattern
            else:
                result = self.application.db.find_all(text)
            num_pages = (len(result) - 1) / page_items + 1;
            if page < 0: page = 0
            if page >= num_pages: page = num_pages - 1
            start = page * page_items 
            start_page = page - 5
            if start_page < 0: start_page = 0
            end_page = page + 5
            if end_page > num_pages: end_page = num_pages - 1
        #print start_page, page, end_page
        self.render("static/admin/admin-search.html", 
                    num_connection = num_connection, 
                    result = result[start: start+page_items],
                    text = text,
                    page_items = page_items,
                    page = page,
                    num_pages = num_pages,
                    start_page = start_page,
                    end_page = end_page)

class RestartHandler(tornado.web.RequestHandler):
    def get(self):
        ip = self.get_argument('ip')
        port = int(self.get_argument('port'))
        if (ip, port) in self.application.decoders_info:
            logging.info("RestartHandler: add decoder(%s, %d) back" % (ip, port))
            self.application.decoders[(ip,port)] = self.application.decoders_info[(ip,port)]
        self.redirect('/admin')


#store wav as files
class WavWebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        logging.debug('new wav arrived %d' % self.application.counter)
        self.wav_name = str(int(time.time())) + '.wav'
        self.wav_data = ''
        self.application.counter += 1
        #self.message_cnt = 0

    def on_message(self, message):
        if message == 'EOS': 
            self.close()
        else :
            #logging.debug('new audio data arrived len %d' % len(message)) 
            self.wav_data += message

    def on_close(self):
        fid = wave.open(self.wav_name, 'wb')
        fid.setnchannels(1)
        fid.setsampwidth(2)
        fid.setframerate(16000)
        fid.writeframes(self.wav_data)
        fid.close()
        logging.debug('write new wav file %s' % self.wav_name)

#decode wav
class DecodeWebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, orgin):
        return True

    def open(self):
#        print self.get_argument('content-type')
#        print self.get_query_argument('content-type')
#        print self.get_query_arguments('content-type')
#        print self.decode_argument(self.get_query_argument('content-type'), 'layout')
#        print self.request
#        print self.request.remote_ip
#        print self.request.headers
#        print self.request.headers['User-Agent']
        try:
            self.user_agent = self.request.headers['User-Agent']
        except KeyError as name_err:
            logging.warning("DecoderWebsocketHandler: No 'User-Agent' key")
            self.user_agent = 'Unknown' # or str(self.application.counter)
            #self.connection_info = (self.request.remote_ip, self.application.counter, time.time())

        try:
            self.phone_type = self.request.headers['Phone-Type']
            self.iesi = self.request.headers['Iesi']
            self.phone_brand = self.request.headers['Phone-Brand']
            self.imei = self.request.headers['Imei']
            self.phone_mac = self.request.headers['Phone-Mac']
        except KeyError as name_err:
            logging.warning("DecoderWebsocketHandler: No 'User-Agent' key")
            self.phone_type = 'null'
            self.iesi = 'null'
            self.phone_brand = 'null'
            self.imei = 'null'
            self.phone_mac = 'null'
            
        self.connection_info = (self.request.remote_ip, self.user_agent, time.time()) 
        self.decoder_error = False
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file_name_base = time.strftime("%Y-%m-%d-%H-%M-%S-") + '%d' % self.application.counter
        self.wav_data = ''
        self.wav_name = os.path.join(self.application.user_data_dir, self.file_name_base + '.wav') 
        self.lab_name = os.path.join(self.application.user_data_dir, self.file_name_base + '.lab')

        max_thread, decoder = 0, None
        with self.application.decoders_list_lock:
            self.application.num_connection += 1
            self.application.connections.add(self.connection_info)
            for key, value in self.application.decoders.items():
                if value > max_thread:
                    max_thread = value
                    decoder = key
            if max_thread <= 0 or decoder == None:
                self.decoder_error = True
            else:
                self.application.decoders[decoder] -= 1
                self.application.counter += 1
        
        if self.decoder_error == True:
            logging.warning("Decoding service is not available now. All decoders are busy now.")
            self.write_message('{"status":9}')
        else:
            self.decoder_info = decoder
            ip, port = self.decoder_info
            logging.debug("selected decoder %s %d\tleft threads:%d" % (ip, port, max_thread-1))
            self.decoder = DecoderClient(self)
            
    def on_message(self, message):
        if message == 'EOS': 
            self.decoder.send_eos()
        else:
            logging.debug('new audio data arrived len %d' % len(message)) 
            assert(len(message)%2 == 0)
            self.decoder.send_wav_data(message)
            self.wav_data += message

    def on_close(self):
        #TODO handle client closed connection
        logging.info("websocket close")
        #none decoder error occured
        with self.application.decoders_list_lock:
            self.application.num_connection -= 1
            assert(self.connection_info in self.application.connections)
            self.application.connections.remove(self.connection_info)
        if not self.decoder_error: 
            try:
                self.decoder.sock.shutdown(SHUT_RDWR)
            except socket.error as err:
                logging.warning('Socket shutdown error')
            with self.application.decoders_list_lock:
                self.application.decoders[self.decoder_info] += 1
            #TODO condition store wav file, eg len(wav_data) > 1s
            
            store_wav_name = 'null'
            if len(self.decoder.final_results) > 0:
                self.write_wav_file(self.wav_name, self.wav_data)
                self.write_label_file(self.lab_name, self.decoder.final_results)
                store_wav_name = self.wav_name
           
            self.application.db.insert(self.start_time, 
                                       self.request.remote_ip, 
                                       self.user_agent,
                                       self.phone_type,
                                       self.iesi,
                                       self.phone_brand,
                                       self.imei,
                                       self.phone_mac,
                                       store_wav_name,
                                       duration=int(time.time() - self.connection_info[2]))

    def write_wav_file(self, file_name, data):
        fid = wave.open(file_name, 'wb')
        fid.setnchannels(1)
        fid.setsampwidth(2)
        fid.setframerate(16000)
        fid.writeframes(data)
        fid.close()
        logging.info('write new wav file %s' % file_name)
    
    def write_label_file(self, file_name, labels):
        with open(file_name, 'w') as fid:
            for label in labels:
                fid.write(label + " : ");

    def handle_decoder_error_with_lock(self):
        #decoder error
        self.write_message('{"status": 8}')
        with self.application.decoders_list_lock:
            del self.application.decoders[self.decoder_info]
        self.decoder_error = True
    
class Application(tornado.web.Application):
    def __init__(self):
        settings = {
            "static_path": os.path.join(os.path.dirname(__file__), "static"),
        }
        handlers = [
            (r'/', IndexHandler),
            (r'/admin', AdminHandler),
            (r'/admin/connections', AdminConnectionsHandler),
            (r'/admin/logs', AdminLogHandler),
            (r'/admin/history', AdminHistoryHandler),
            (r'/admin/search', AdminSearchHandler),
            (r'/restart', RestartHandler),
            (r'/ws/speech', WavWebSocketHandler),
            (r'/ws/decode', DecodeWebSocketHandler),
            (r'/static/', tornado.web.StaticFileHandler, dict(path=settings['static_path'])),
        ]
        tornado.web.Application.__init__(self, handlers, **settings)
        self.counter = 0
        self.num_connection = 0
        self.connections = set()
        self.decoders = {}
        self.decoders_info = {}
        self.decoders_list_lock = threading.Lock()
        self.load_config('main.cfg')

    def load_config(self, config_file):
        #TODO, add other config item
        config = ConfigParser.ConfigParser()
        config.read(config_file)
        # config server port
        self.port = config.getint('master', 'port')
        # config log
        self.log_file = config.get('master', 'log_file')
        logging.basicConfig(level = logging.DEBUG, 
                            format = '%(levelname)s %(asctime)s (%(filename)s:%(funcName)s():%(lineno)d) %(message)s', 
                            filename = self.log_file, 
                            filemode = 'a')
        # config decoder cluster
        for item in config.items('decoders'):
            #check ip and port
            arr = item[1].split(':')
            assert(len(arr) == 3)
            ip, port, max_thread = arr[0], int(arr[1]), int(arr[2])
            assert(port > 0 and port < 65535)
            logging.debug("add decoder " + item[1])
            self.decoders[(ip, port)] = max_thread
            self.decoders_info[(ip, port)] = max_thread
        assert(len(self.decoders) > 0)
        # config database
        db_file =  config.get('master', 'database')
        self.db = DataBase(db_file)
        # config user-data
        self.user_data_dir = config.get('master', 'user_data_dir')
        if not os.path.exists(self.user_data_dir):
            os.makedirs(self.user_data_dir)
    
    def start(self):
        logging.info('server start, port: %d...' % self.port)
        try:
            self.listen(self.port)
            tornado.ioloop.IOLoop.current().start()
        except:
            self.db.close()

if __name__ == "__main__":
    Application().start()
