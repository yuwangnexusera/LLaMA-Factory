import sys
sys.path.append(".")
from utils.ds_label import ds_image


def get_obj_by_id(id):
    query = ds_image.get_by_id(id)
    return query

def select_ds():
    query = ds_image.select()
    return query
if __name__=='__main__':
    print(select_ds())
