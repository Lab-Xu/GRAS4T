# -*- coding: utf-8 -*-

def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

        print(path + '| Path created successfully!')
        return True
    else:
        print(path + '| Path already exists...')
        return False
