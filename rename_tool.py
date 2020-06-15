# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:19:34 2020

a little tool to rename multitudinous files

@author: Brynhildr
"""

#%% import modules
import os
import sys
import re
 
#%% define basic functions
def rename():
    path = input('请输入文件路径: ')
    ques_mode = '请选择命名模式（1或2）：\n'
    ques_mode += '1-快速模式（即xxx01）\n'
    ques_mode += '2-高级模式（针对qq音乐文件）\n：'
    mode = int(input(ques_mode))

    if mode == 1:  # easy mode
        # initialization
        iniName = input('请输入起始名：')
        iniNum = input('请输入起始编号：')
        fileType = input('请输入后缀名（如.mp3、.txt、.jpg等）: ')
        print('批量操作中...')
        # operation
        fnum = 0  # number of files
        filelist = os.listdir(path)  # load all files' names
        for files in filelist:
            old_dir = os.path.join(path, files)
            if os.path.isdir(old_dir):
                continue
            new_dir = os.path.join(path, iniName + str(fnum + int(iniNum)) + fileType)
            os.rename(old_dir, new_dir)
            fnum += 1
        print('已完成' + str(fnum) + '个文件')

    elif mode == 2:  # advanced mode
        # initialization
        print('本模式需要预先读取原始文件名，之后按需重设')
        # config str to be deleted
        my_regex = '\(.*\)|\s-\s.*'  # any str inside ()
        # operation
        filelist = os.listdir(path)
        group = input('文件名读取完成，输入专辑/电影名称：')
        group = '【' + group + '】'
        print('批量操作中...')
        fnum = 0
        for files in filelist:
            # extract author name & deletion
            temp = files.split(' - ')  # author & title & type
            fileAuthor = re.sub(my_regex, '', temp[0]).strip().replace(' _ ', '、')              
            fileName = re.sub(my_regex, '', temp[1].split('.')[0]).strip()
            fileType = '.' + temp[1].split('.')[-1]
            # rename
            old_dir = os.path.join(path, files)
            if os.path.isdir(old_dir):
                continue
            new_dir = os.path.join(path, group + fileAuthor + ' - ' + fileName
                               + fileType)
            os.rename(old_dir, new_dir)
            fnum += 1
        print('已完成' + str(fnum) + '个文件')
            
#%% operation
rename()