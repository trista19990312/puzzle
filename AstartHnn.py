
import numpy as np
import copy
import time
import os

from NetworkHnn10 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def Manhattan_distance(node_a, node_b):
    N = node_b.N
    a2 = node_a.nodes.reshape((N, N))
    b2 = node_b.nodes.reshape((N, N))

    return Manhattan_distance_array(a2, b2)


def Manhattan_distance_array(array_a, array_b):
    shape = np.shape(array_a)
    N = shape[0]
    N2 = N*N

    ap = np.zeros(shape=(N2, 2), dtype='int16')
    bp = np.zeros(shape=(N2, 2), dtype='int16')

    for y in range(N):
        for x in range(N):
            ap[array_a[y, x], :] = [x, y]
            bp[array_b[y, x], :] = [x, y]

    return np.sum(np.abs((ap - bp)[0:N2-1, :]))*1.625    #*1.5


def node_extend(father_node):
    a_list = []

    a_u = copy.deepcopy(father_node)
    if a_u.up() is not None:
        a_u.father = father_node
        a_list.append(a_u)

    a_d = copy.deepcopy(father_node)
    if a_d.down() is not None:
        a_d.father = father_node
        a_list.append(a_d)

    a_l = copy.deepcopy(father_node)
    if a_l.left() is not None:
        a_l.father = father_node
        a_list.append(a_l)

    a_r = copy.deepcopy(father_node)
    if a_r.right() is not None:
        a_r.father = father_node
        a_list.append(a_r)

    return a_list


def in_open_close(neighbor, open_close_list):
    """
    open & close队列中重复节点检查
    """
    in_list = []

    for idx in range(len(open_close_list)):
        oc_item = open_close_list[idx]
        if neighbor.x != oc_item.x:        # speed up
            continue

        if neighbor.y != oc_item.y:        # speed up
            continue

        if np.array_equal(neighbor.nodes, oc_item.nodes) is False:
            continue

        in_list.append(idx)
        break

    return in_list


def states_save(filename, states_list):
    import pickle as pk

    # 生成pkl格式的文件，保存数据。
    print('磁盘写入中...\n')
    f_out = gzip.open(filename + '.pkl.gz', 'wb')
    pk.dump(states_list, f_out, -1)
    f_out.close()

    import winsound
    winsound.Beep(300, 1500)     # 其中300表示声音大小，1500表示发生时长，1000为1秒


def states_save_auto(states_list):
    import os

    kk = 1
    sn_str = str(kk).zfill(3)
    file_sn_name = sn_str + '.pkl.gz'
    while os.path.isfile(file_sn_name) is True:
        kk += 1
        sn_str = str(kk).zfill(2)
        file_sn_name = sn_str + '.pkl.gz'

    # 生成pkl格式的文件，保存数据。
    states_save(sn_str, states_list)


class State(object):
    def __init__(self, N=4, father=None):
        self.N = N
        self.NN = N*N
        self.shape = (N, N)
        self.father = father

        self.nodes_1d = np.array([x for x in range(self.NN)], dtype='int16')

        random_method = np.random.RandomState(int(time.time()))
        random_method.shuffle(self.nodes_1d[0:self.NN-1])

        #self.nodes_1d = np.array([0,1,2,3,4,6,7,11,8,5,13,10,12,9,14,15], dtype='int16')
        #self.nodes_1d = np.array([1,5,3,7,0, 6,2,11,4,13,9,14,8,12,10,15], dtype='int16')
        self.nodes_1d = np.array([0,3,8,6,14,5,9,7,13,4,11,10,12,2,1,15], dtype='int16')
        #self.nodes_1d = np.array([9,4,0,6,2,8,3,13,5,1,11,10,12,7,14,15], dtype='int16')
        #self.nodes_1d = np.array([4,3,7,6,1,14,5,10,12,11,9,2,8,0,13,15], dtype='int16')
        #self.nodes_1d = np.array([6,12,4,7,10,3,1,2,5,9,14,13,0,8,11,15], dtype='int16')

        while self.is_no_solution():
            random_method = np.random.RandomState(int(time.time()))
            random_method.shuffle(self.nodes_1d[0:self.NN-1])

        self.nodes = self.nodes_1d.reshape(self.shape)

        self.y, self.x = tuple(np.argwhere(self.nodes == self.NN-1)[0])
        self.gn = 0
        self.hn = -1
        self.fn = 0

        self.dst_state = np.array([x for x in range(self.NN)], dtype='int16')
        self.dst_state = self.dst_state.reshape(self.shape)

    def up(self):
        if self.y > 0 and (self.father is None or self.father.y != self.y-1):
            self.nodes[self.y, self.x] = self.nodes[self.y-1, self.x]
            self.y -= 1
            self.nodes[self.y, self.x] = self.NN-1

            self.gn += 1
            return self.y, self.x
        else:
            return None

    def down(self):
        if self.y < self.N-1 and (self.father is None or self.father.y != self.y+1):
            self.nodes[self.y, self.x] = self.nodes[self.y+1, self.x]
            self.y += 1
            self.nodes[self.y, self.x] = self.NN-1

            self.gn += 1
            return self.y, self.x
        else:
            return None

    def left(self):
        if self.x > 0 and (self.father is None or self.father.x != self.x-1):
            self.nodes[self.y, self.x] = self.nodes[self.y, self.x-1]
            self.x -= 1
            self.nodes[self.y, self.x] = self.NN-1

            self.gn += 1
            return self.y, self.x
        else:
            return None

    def right(self):
        if self.x < self.N-1 and (self.father is None or self.father.x != self.x+1):
            self.nodes[self.y, self.x] = self.nodes[self.y, self.x+1]
            self.x += 1
            self.nodes[self.y, self.x] = self.NN-1

            self.gn += 1
            return self.y, self.x
        else:
            return None

    def gn_hn(self):
        self.hn = Manhattan_distance_array(self.nodes, self.dst_state)
        self.fn = self.gn + self.hn

    def hnn(self, hnn_net):
        self.hn = hnn_net.hnn(self.nodes.flatten())[0]  # TypeError: unsupported operand type(s) for +: 'int' and 'list'
        self.fn = self.gn + self.hn

    def is_no_solution(self):
        position = np.zeros(shape=(self.NN, ), dtype='int16')
        for p in range(self.NN):
            d = self.nodes_1d[p]
            position[d] = p

        sn = 0
        for j in range(1, self.NN-1):
            pj = position[j]
            for i in range(j):
                if position[i] > pj:
                    sn += 1

        if (sn % 2) == 0:
            return False
        else:
            return True


if __name__ == '__main__':
    net = Network([
                    FullyConnectedLayer(n_in=16, n_out=32),
                    FullyConnectedLayer(n_in=32, n_out=32),
                    LinearLayer(n_in=32, n_out=1)
                   ])
    net.load_model()

    # 目标状态
    goal_state = State()
    goal_state.nodes = np.array([x for x in range(goal_state.NN)], dtype='int16')
    goal_state.nodes = goal_state.nodes.reshape(goal_state.shape)

    t0 = time.time()
    for k in range(1):
        # 清空open & close
        open_list = []
        close_list = []

        # 任意可解的初始状态, 加入open
        src_state = State()
        open_list.append(src_state)

        for i in range(50000):
            # pop出open队列首节点
            head_state = open_list.pop(0)

            # 已删除节点标识
            if head_state.fn < 0:
                continue

            # open头节点加入close
            close_list.append(head_state)

            m_dist = Manhattan_distance(goal_state, head_state)
            if i % 1 == 0:
                print(k, i, '\t', head_state.gn, '\t', head_state.hn, '\t', head_state.fn, '\t', len(open_list))

            if m_dist == 0:
                path_list = []
                t = head_state
                data = (t.nodes.flatten())
                path_list.append(data)
                print(t.nodes)

                while t.father is not None:
                    t = t.father
                    data = (t.nodes.flatten())
                    path_list.append(data)
                    print(t.nodes)

                #states_save_auto(path_list)

                break

            # 扩展节点
            expend4 = node_extend(head_state)

            # 扩展节点检测重复 & 加入open
            if len(expend4) == 0:
                continue

            for item in expend4:
                # in open?
                in_open_l = in_open_close(item, open_list)
                if len(in_open_l) != 0:
                    item_open = open_list[in_open_l[0]]
                    if item.gn < item_open.gn:
                        del open_list[in_open_l[0]]
                        #open_list[in_open_l[0]].fn = -1
                        #open_list[in_open_l[0]].x = -1
                    else:
                        continue

                # in close?
                in_close_l = in_open_close(item, close_list)
                if len(in_close_l) != 0:
                    item_close = close_list[in_close_l[0]]
                    if item.gn < item_close.gn:
                        del close_list[in_close_l[0]]
                    else:
                        continue

                # not in open & close
                item.hnn(hnn_net=net)       # h(n)=hnn
                #item.gn_hn()               # h(n)=MD
                open_list.append(item)

            # open重排序
            def take_fn(elem):
                return elem.fn
            open_list.sort(key=take_fn)

        t1 = time.time()
        print(t1-t0)

    exit(0)
