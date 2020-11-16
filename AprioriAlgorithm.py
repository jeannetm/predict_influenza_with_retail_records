# Machine learning and Data Mining - Association Analysis with Python
# http://aimotion.blogspot.it/2013/01/machine-learning-and-data-mining.html

# While the number of items in the set is greater than 0:
#   Create a list of candidate itemsets of length k
#   Scan the dataset to see if each itemset is frequent
#   Keep frequent itemsets to create itemsets of length k+1


def createC1(dataset):
    # Create a list of candidate item sets of size one.
    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    # frozenset because it will be a key of a dictionary
    return map(frozenset, c1)


def scanD(dataset, candidates, min_support):
    # Returns all candidates that meets a minimum support level
    sscnt = {}
    for tid in dataset:
        for can in candidates:
            if can.issubset(tid):
                sscnt.setdefault(can, 0)
                sscnt[can] += 1
    num_items = float(len(dataset))
    retlist = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key] / num_items
        if support >= min_support:
            retlist.insert(0, key)
        support_data[key] = support
    return retlist, support_data


def aprioriGen(freq_sets, k):
    # Generate the joint transactions from candidate sets
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(freq_sets[i] | freq_sets[j])
    return retList


def apriori(dataset, minsupport):
    # Generate a list of candidate item sets
    C1 = createC1(dataset)  # We create the first candidate item set C1. C1 contains a list of all items in frozenset.
    D = map(set, dataset)  # We create D, which is just a dataset in the form of list of sets (set is a list of unique elements).
    L1, support_data = scanD(D, C1, minsupport)  # With everything in set form we remove items that don't meet the minimum support.
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)  # AprioriGen() is responsible for generate the next list of candidate itemsets
        Lk, supK = scanD(D, Ck, minsupport)
        support_data.update(supK)  # The variable support_data is just a dictionary with the support values of our itemsets.
        L.append(Lk)
        k += 1
    return L, support_data
