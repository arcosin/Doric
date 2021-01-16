

I_FUNCTION = (lambda x : x)
BN_FILTER = (lambda n: ("BN" not in n))


def enumerateFloatTensor(tensor, indexList = []):
    if tensor.shape != torch.Size([]):
        ret = []
        for ind, t, in enumerate(tensor):
            ret = ret + enumerateFloatTensor(t, indexList + [ind])
        return ret
    else:
        return [(indexList, tensor.item())]


def getNBiggestWeights(mod, n = 10, biggest = True, absVal = True, nameFilter = I_FUNCTION):
    paramList = []
    for paramName, paramTensor in mod.named_parameters():
        if nameFilter(paramName):
            for indexList, fl in enumerateFloatTensor(paramTensor):
                if absVal:
                    paramList.append((abs(fl), paramName + "_" + str(indexList).replace(" ", "")))
                else:
                    paramList.append((fl, paramName + "_" + str(indexList).replace(" ", "")))
    paramList.sort(biggest)
    return paramList[:n]


def zeroLaterals(prognet, val = 0.0):
    stateDict = prognet.state_dict()
    for paramName, paramTensor in stateDict.items():
        if "laterals" in paramName:
            filledTensor = paramTensor.fill_(val)
            stateDict[paramName] = filledTensor
    prognet.load_state_dict(stateDict)


#===============================================================================
