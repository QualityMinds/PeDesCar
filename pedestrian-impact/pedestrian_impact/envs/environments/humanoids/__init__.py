from .atlas import Atlas
from .talos import Talos
from .unitreeH1 import UnitreeH1
from .humanoids import HumanCollisionTorque, HumanCollisionMuscle, HumanCollisionTorque4Ages, HumanCollisionMuscle4Ages


# register environments in mushroom
Atlas.register()
Talos.register()
UnitreeH1.register()
HumanCollisionTorque.register()
HumanCollisionMuscle.register()
HumanCollisionTorque4Ages.register()
HumanCollisionMuscle4Ages.register()


from gymnasium import register

# register gymnasium wrapper environment
register("HumanCollision",
         entry_point="environments.gymnasium:GymnasiumWrapper"
         )
