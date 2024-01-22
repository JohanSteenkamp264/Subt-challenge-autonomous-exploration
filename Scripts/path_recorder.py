import numpy as np
import os


class pose_node:
    def __init__(self, current_pose, node_index, parent_node=None, is_exploration_node=False):
        self.node_index = node_index
        self.pose = current_pose
        self.parent_node = parent_node
        self.exploration_node = is_exploration_node
        self.child_nodes = []

    def add_child_node(self, node):
        if node != None:
            self.child_nodes.append(node)

    def have_exploration_child(self):
        for node in self.child_nodes:
            if node.exploration_node:
                return True

        return False


class path_recorder:
    def __init__(self):
        self.THR_LOOP_CLOSE = 2.0
        self.MIN_INDEX_DIFFERENCE = 240

        self.INDEX_DIFFERENCE_DISTANCE_FROM_PATH = 240

        self.root_node = None
        self.current_node = None

        self.exploration_nodes = []
        self.positional_nodes = []
        self.previous_poses = None

    def add_position_node(self, pose):
        if self.root_node == None:
            self.root_node = pose_node(pose, node_index=len(self.positional_nodes))
            self.current_node = self.root_node

            self.positional_nodes.append(self.root_node)
            self.previous_poses = np.array(np.array([pose]), dtype=np.float32)
        else:
            self.previous_poses = np.append(self.previous_poses, [pose], axis=0)

            node = pose_node(pose, node_index=len(self.positional_nodes), parent_node=self.current_node)


            self.current_node.add_child_node(node)
            self.current_node = node
            self.positional_nodes.append(node)

            return

    def add_exploration_node(self, predicted_pose):
        if self.root_node == None:
            raise Exception("unable to add exploration node to empty path")
        else:
            for ex_node in self.exploration_nodes:
                dist = np.sqrt(np.sum(np.square(ex_node.pose - predicted_pose)))
                if dist < 15.0:
                    print("exploration node not added due to distance to other nodes")
                    return

            node = pose_node(predicted_pose, node_index= None, parent_node=self.current_node, is_exploration_node=True)
            self.current_node.add_child_node(node)
            self.exploration_nodes.append(node)

    def is_similar_path(self, pos_1, pos_2, tolerance=10.0):
        pos_1_local_translation = self.get_local_translation(pos_1)
        dist_pos_1 = np.sqrt(np.sum(np.square(pos_1_local_translation)))

        pos_2_local_translation = self.get_local_translation(pos_2)
        dist_pos_2 = np.sqrt(np.sum(np.square(pos_2_local_translation)))

        dist = min(dist_pos_1, dist_pos_2)

        pos_1_local_translation = (dist / dist_pos_1) * pos_1_local_translation
        pos_2_local_translation = (dist / dist_pos_2) * pos_2_local_translation

        distance_between_paths = np.sqrt(np.sum(np.square(pos_1_local_translation - pos_2_local_translation)))
        return distance_between_paths < tolerance

    def get_local_translation(self, pose):
        global_translation = np.array(pose - self.get_current_pose())
        return global_translation

    def get_current_pose(self):
        return self.current_node.pose

    def get_closest_node_position(self, pose, threshold = 1.5):
        if len(self.previous_poses) > self.INDEX_DIFFERENCE_DISTANCE_FROM_PATH:
            distances = np.sqrt(np.sum(np.square(pose - self.previous_poses[0:len(self.previous_poses) - self.INDEX_DIFFERENCE_DISTANCE_FROM_PATH]), axis=1))

            return self.previous_poses[np.argwhere(distances < threshold)]
        else:
            return []

    def has_path_been_explored(self, pose, threshold = 5.0):
        return self.get_closest_distance_in_map(pose) < threshold

    def get_closest_distance_in_map(self, pose):
        if np.all(self.previous_poses) != None and len(self.previous_poses) > self.INDEX_DIFFERENCE_DISTANCE_FROM_PATH:
            distances = np.sqrt(np.sum(np.square(pose - self.previous_poses[0:len(self.previous_poses) - self.INDEX_DIFFERENCE_DISTANCE_FROM_PATH]), axis=1))
            return np.min(distances)
        else:
            return np.float("inf")

    def get_transverse_path_to_closest_exploration_node(self, num_skips = 0, return_to_start = False):
        for index in reversed(range(len(self.exploration_nodes))):
            ex_node = self.exploration_nodes[index]
            if self.get_closest_distance_in_map(ex_node.pose) < 10.0:
                ex_node.exploration_node = False
                print("node {} Found to be explored".format(ex_node.pose))
                del self.exploration_nodes[index]

        transverse_path = []
        '''distances = np.sqrt(np.sum(np.square(self.current_node.pose - self.previous_poses), axis=1))
        indexes = np.argwhere(distances < self.THR_LOOP_CLOSE)'''

        transverse_node = self.current_node
        skips = 0
        while transverse_node != None:
            transverse_path.append(transverse_node.pose)

            if transverse_node.have_exploration_child() and not return_to_start:
                for node in transverse_node.child_nodes:
                    if node.exploration_node:
                        if skips >= num_skips:
                            transverse_path.append(node.pose)
                            return transverse_path
                        else:
                            skips += 1
                            transverse_node = transverse_node.parent_node

            else:
                transverse_node = transverse_node.parent_node

        return transverse_path

    def display_exploration_nodes(self):
        print("Exploration nodes:")
        for node in self.exploration_nodes:
            print("\tx: {:.2f}\ty: {:.2f}\tz: {:.2f} E:{}".format(node.pose[0], node.pose[1], node.pose[2], node.exploration_node))

    def save_map_to_file(self, file_path):
        if os.path.isfile(file_path):
            os.remove(file_path)

        node = self.current_node
        nodes = []
        while node != None:
            nodes.append(node)
            node = node.parent_node

        f_map = open(file_path, 'w')
        for nde in reversed(nodes):
            f_map.write("{},{},{}\n".format(nde.pose[0], nde.pose[1], nde.pose[2]))
            for c_nde in nde.child_nodes:
                if c_nde.exploration_node:
                    f_map.write("#{},{},{}\n".format(c_nde.pose[0], c_nde.pose[1], c_nde.pose[2]))

    def get_max_dist_in_map(self,pose):
        if np.any(self.previous_poses == None):
            return np.sqrt(np.sum(np.square(pose - self.previous_poses), axis=1))
        else:
            return np.float("inf")



def get_path_recorder_from_file(file_path):
    if os.path.isfile(file_path):
        map = path_recorder()
        for line in open(file_path).read().split("\n"):
            if line != "":
                sp_line = line.replace("#","").split(',')
                pose = np.array([float(sp_line[0]), float(sp_line[1]), float(sp_line[2])])
                if line[0] == '#':
                    # Exploration node
                    map.add_exploration_node(pose)
                else:
                    # Position node
                    map.add_position_node(pose)

        return map
    else:
        return path_recorder()

