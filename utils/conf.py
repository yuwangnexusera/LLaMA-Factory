import datetime
import os
# mysql
db_conf = {
    "host": "192.168.19.134",
    "port": 3306,
    "database": "med_books2",
    "username": "root",
    "password": "1qaz2wsX",
}
ALLOWED_IPS = [
    "127.0.0.0/24",
    "192.168.19.0/24",
    "192.168.19.240",
    "172.17.151.11",
    "182.92.119.176",
    "10.200.10.0/24",
    "10.200.100.0/24",
    "10.19.53.208",
]
