import rospy
import numpy as np
from numpy.typing import NDArray
from collections import deque
from typing import List
from std_msgs.msg import Float64MultiArray


class Gossip:
    class NeighborSubscriber:
        def __init__(self, neighbor_namespace: str, topic_name: str):
            self.subscriber = rospy.Subscriber(
                neighbor_namespace + "/" + topic_name,
                Float64MultiArray,
                self.callback,
                queue_size=10,
            )

            # Here, we use a buffer of size 10 to store the received messages.
            # This is because the subscriber callback is automaticly called in another thread and we can't control it.
            # This part is different from the C++ version.
            self._deque = deque(maxlen=10)

        def callback(self, message: Float64MultiArray):
            self._deque.append(message.data)

        def empty(self) -> bool:
            return len(self._deque) == 0

        def receive(self) -> NDArray[np.float64]:
            data_lst = self._deque.popleft()
            data = np.array(data_lst, dtype=np.float64)

            return data

    def __init__(
        self, self_namespace: str, neighbor_namespaces: List[str], topic_name: str
    ):
        self._publisher = rospy.Publisher(
            self_namespace + topic_name, Float64MultiArray, queue_size=10
        )
        self._message = Float64MultiArray()

        self._subscribers = [
            self.NeighborSubscriber(neighbor_namespace, topic_name)
            for neighbor_namespace in neighbor_namespaces
        ]

    def broadcast(self, state: NDArray[np.float64]):
        self._message.data = state.tolist()
        self._publisher.publish(self._message)

    def gather(self) -> List[NDArray[np.float64]]:
        neighbor_states = []

        for subscriber in self._subscribers:
            if not subscriber.empty():
                neighbor_states.append(subscriber.receive())

        return neighbor_states

    def calculate_consensus_error(
        self, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self.broadcast(state)
        neighbor_states = self.gather()

        consensus_error = len(neighbor_states) * state - sum(neighbor_states)

        return consensus_error
