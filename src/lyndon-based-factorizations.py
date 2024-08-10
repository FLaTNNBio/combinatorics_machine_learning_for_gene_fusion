
# ------------------------ CFL ---------------------------------------------------------------------
# CFL - Lyndon factorization - Duval's algorithm
def CFL(word,T):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0
    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or word[j - 1] < word[i - 1]:
                while k < i:
                    # print(word[k:k + j - i])
                    CFL_list.append(word[k:k + j - i])
                    k = k + j - i
                break
            else:
                # if word[j-1] > word[i-1]:
                if word[j - 1] > word[i - 1]:
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1

    return CFL_list


def contains(window, preview, next_c):
    """Check if [window + preview] contains [preview + next_c]"""
    return ''.join(preview + [next_c]) in ''.join(window + preview)


def find_index(window, preview):
    """Return the first index of preview occurring in window
       relative to the window size"""
    dictionary = ''.join(window + preview)
    return len(window) - dictionary.find(''.join(preview))


# ----------------------- ICFL ----------------------------------------------------------------------
# ICFL recursive (without using of compute_br)- Inverse Lyndon factorization
def ICFL_recursive(word):
    """In this version of ICFL, we don't execute compute_br - one only O(n) scanning of word"""
    br_list = []
    icfl_list = []

    compute_icfl_recursive(word, br_list, icfl_list)

    return icfl_list

def compute_icfl_recursive(word, br_list, icfl_list):

    # At each step compute the current bre
    pre_pair = find_pre(word)
    current_bre_quad = find_bre(pre_pair)
    br_list.append(current_bre_quad)

    if current_bre_quad[1] == '' and current_bre_quad[0].find('$') >= 0:
        w = current_bre_quad[0]
        icfl_list.insert(0, w[:len(w) - 1])
        return
    else:
        compute_icfl_recursive(current_bre_quad[1] + current_bre_quad[2], br_list, icfl_list)
        if len(icfl_list[0]) > current_bre_quad[3]:
            icfl_list.insert(0, current_bre_quad[0])
        else:
            icfl_list[0] = current_bre_quad[0] + icfl_list[0]
        return


def find_pre(word):
    if len(word) == 1:
        return (word + "$", '')
    else:
        i = 0
        j = 1

        while j < len(word) and word[j] <= word[i]:
            if word[j] < word[i]:
                i = 0
            else:
                i = i + 1
            j = j + 1

        if j == len(word):
            return (word + "$", '')
        else:
            return (word[0:j + 1], word[j + 1:len(word)])


def find_bre(pre_pair):
    w = pre_pair[0]
    v = pre_pair[1]

    if v == '' and w.find('$') >= 0:
        #return (w[:len(w)-1], '', '', 0)
        return (w, '', '', 0)
    else:
        n = len(w) - 1

        f = border(w[:n])

        i = n
        last = f[i-1]

        while i > 0:
            if w[f[i-1]] < w[n]:
                last = f[i-1]
            i = f[i-1]

        return (w[:n - last], w[n - last:n + 1], v, last)


def border(p):
    l = len(p)
    pi = [0]
    k = 0
    for i in range(1, l):
        while (k > 0 and p[k] != p[i]):
            k = pi[k-1]
        if (p[k] == p[i]):
            pi.append(k +1)
            k = k + 1
        else:
            pi.append(k)

    return pi


def compute_br(w, br_list):
    pre_pair = find_pre(w)
    bre_quad = find_bre(pre_pair)

    if bre_quad[1] == '':
        br_list.append(bre_quad)
        return br_list
    else:
        br_list.append(bre_quad)
        compute_br(bre_quad[1] + bre_quad[2], br_list)


# ------------------------ CFL_icfl ---------------------------------------------------------------------
# CFL factorization - ICFL subdecomposition
def CFL_icfl(word,C):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0
    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or word[j - 1] < word[i - 1]:
                while k < i:
                    # print(word[k:k + j - i])
                    w = word[k:k + j - i]
                    if len(w) <= C:
                        CFL_list.append(word[k:k + j - i])
                    else:
                        ICFL_list_recursive = ICFL_recursive(w,C)

                        # Insert << to indicate the begin of the subdecomposition of w
                        CFL_list.append("<<")
                        for v in ICFL_list_recursive:
                            CFL_list.append(v)
                        # Insert >> to indicate the end of the subdecomposition of w
                        CFL_list.append(">>")

                    k = k + j - i
                break
            else:
                # if word[j-1] > word[i-1]:
                if word[j - 1] > word[i - 1]:
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1

    return CFL_list


if __name__ == "__main__":

    w = "Hi, Lorraine!"
    icfl_list = ICFL_recursive(w)
    print("icfl - list of factors: {}".format(icfl_list))