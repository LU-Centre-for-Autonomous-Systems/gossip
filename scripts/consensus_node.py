#!/usr/bin/env python3

import rospy
import numpy as np
import gossip
from std_msgs.msg import Float64MultiArray

if __name__ == "__main__":
    rospy.init_node("gossip_test", anonymous=True)

    namespace = rospy.get_namespace()
    neighbor_namespaces = rospy.get_param("~neighbor_namespaces", [])

    state_lst = rospy.get_param("~state", [])
    rate_hz = rospy.get_param("~rate_hz", 10)
    alpha = rospy.get_param("~alpha", 0.3)

    state = np.array(state_lst, dtype=np.float64)
    state_msg = Float64MultiArray()

    comm = gossip.Gossip(namespace, neighbor_namespaces, "consensus_topic")

    state_pub = rospy.Publisher("state", Float64MultiArray, queue_size=10)

    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        state_msg.data = state.tolist()
        state_pub.publish(state_msg)

        error_state = comm.calculate_consensus_error(state)
        state = state - alpha * error_state

        rate.sleep()
