import pymysql
import numpy as np


mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

sql="show tables like 'number%';"
cursor.execute(sql)
table_list = cursor.fetchall()


for i in table_list :
    sql="desc "+i['Tables_in_sa_server (number%)']+";"
    cursor.execute(sql)
    col_list = cursor.fetchall()
    for j in col_list:
        if j['Field'].find('SPO') != -1:
            print(i)
            print(j)