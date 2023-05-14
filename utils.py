def obj2dict(obj, attrlist: list[str]):
    dic = {}
    for a in attrlist:
        dic[a] = getattr(obj, a)
    return dic
