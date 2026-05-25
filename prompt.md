# Prompt

Action Space Incremental Reinforcement Learning
In general reinforcement learning, action space is pre-defined (number of actions are fixed throughout the task). But, in real world situations, we may need to use new actions to accomplish the tasks. It can also be observed in the video games where the number of actions may increase with increase in level of the game.

As an example, consider a well known DAVE game. In level one, there are the actions {up}, {left}, {right}. Once, we enter level three, we will get an additional action of {shooting}. If we do not know that in future we may get new action, one has to learn new action, and may need to train the reinforcement learning algorithm from scratch with the additional action. This will take lot of time and the knowledge gained in previous levels goes waste.

In this thesis, a novel learning technique that can adapt the new action incrementally not from the scratch. The algorithm shall not lose the existing knowledge (ex: knowledge of previous levels in video games).

References:

1. Venkatesan, Rajasekar, and Meng Joo Er. "A novel progressive learning technique for multi-class classification." Neurocomputing 207 (2016): 310-321.
2. Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
