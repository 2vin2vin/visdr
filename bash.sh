#!/bin/bash

echo 'started'
git init
git add *
echo 'added'
git commit -m "first commit"
echo 'commited'
git branch -M main
git remote add origin https://github.com/2vin2vin/visdr.git
git push -u origin main
echo 'pushed'
