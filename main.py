from Utils.enums import User, Environment
from configs import configure

configure(enable_mixed_float16=False,
          print_device_placement=False)

user = User.Arash
# user = User.Kinza

environment = Environment.Local
# environment = Environment.GoogleColab

