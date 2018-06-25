#!/bin/bash


python main.py &
sleep 1
tail -f static/server.log
