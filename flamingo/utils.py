

# obj 
# attr 'model.decoder.layers'
# return obj.model.decoder.layers

def getattr_recursively(obj, attr):
    
    if (attr == ""):
        return obj
    
    idx = attr.find('.')
    
    if (idx < 0):
        return getattr(obj, attr)
    
    return getattr_recursively(getattr(obj, attr[:idx]), attr[idx+1:])

def setattr_recursively(obj, attr, value):
    
    if (attr == ""):
        obj = value
        
    idx = attr.find('.')
    
    if (idx < 0):
        setattr(obj, attr, value)
        return
    
    setattr_recursively(getattr(obj, attr[:idx]), attr[idx+1:], value)