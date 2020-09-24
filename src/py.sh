#!/bin/bash

pip3 -V >> /dev/null

if [ "$?" -eq "0" ]
then

  mode=$1

  case "$mode" in

    "install") pip3 install -r requirements.txt ;;
    "save") pip3 freeze > requirements.txt ;;
    "test") python3 test.py ;;

  esac

else
  printf "Python 3 is needed."
  exit 123
fi
