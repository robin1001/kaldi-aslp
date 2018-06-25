# Created on 2015-10-17
# Author: zhangbinbin 
    
import sqlite3
import os
import threading
import logging


#conn = sqlite3.connect('my.db')
#c = conn.cursor()

# Create table
#create_cmd = '''CREATE TABLE history (id integer primary key autoincrement, 
#                         time datetime not null, 
#                         ip varchar(20) not null, 
#                         user_agent text not null, 
#                         duration integer not null)'''
#c.execute(create_cmd)

# Insert a row of data
#c.execute('''INSERT INTO history (time, ip, user_agent, duration) 
#             VALUES (datetime('now'),'127.0.0.1','Windows NT',100)''')

# Save (commit) the changes
#conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
#conn.close()

#c.execute('select * from history order by id')
#print c.fetchall()

#for x in c.execute('select * from history order by id'):
#    print x

class DataBase:
    def __init__(self, db_file):
        exist =  os.path.exists(db_file)
        self.conn = sqlite3.connect(db_file)
        self.cur = self.conn.cursor()
        self.lock = threading.Lock()
        if not exist:
            sql_cmd = '''CREATE TABLE history 
                            (id integer primary key autoincrement, 
                            time datetime not null, 
                            ip varchar(20) not null, 
                            user_agent text not null,
                            phone_type text not null,
                            iesi text not null,
                            phone_brand text not null,
                            imei text not null,
                            phone_mac text not null,
                            wav_name text not null,
                            duration integer not null)'''
            self.cur.execute(sql_cmd)

    def insert(self, time, ip, user_agent, phone_type='null', iesi='null', phone_brand='null', imei='null', phone_mac='null', wav_name='null', duration=0):
        with self.lock:
            sql_cmd = '''INSERT INTO history (time, ip, user_agent, phone_type, iesi, phone_brand, imei, phone_mac, wav_name, duration) 
                         VALUES (datetime(?), ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            self.cur.execute(sql_cmd, (time, ip, user_agent, phone_type, iesi, phone_brand, imei, phone_mac, wav_name, duration))
            self.conn.commit()

    def get_all(self): 
        sql_cmd = '''select * from history order by id desc'''
        return self.cur.execute(sql_cmd).fetchall()

    def find_all(self, pattern):
        sql_cmd = 'select * from history where ' +  \
                              'time like \'%' + pattern + '%\' or ' + \
                              'ip like \'%' + pattern + '%\' or ' + \
                              'user_agent like \'%' + pattern + '%\' order by id desc' 
        #res = self.cur.execute(sql_cmd, (pattern, pattern, pattern))
        logging.info('sql cmd >> ' + sql_cmd)
        res = self.cur.execute(sql_cmd)
        return res.fetchall()

    def execute_sql(self, sql):
        with self.lock:
            return self.cur.execute(sql).fetchall()

    def close(self):
        self.conn.close()

if __name__ == '__main__':
    db = DataBase('history.db') 
    db.insert('2015-10-12', '127.0.0.1', 'Windows NT', duration=100)
    db.insert('2015-10-13 12:20:35', '127.0.0.1', 'Windows NT', duration=60)
    print db.get_all()
    db.close()
