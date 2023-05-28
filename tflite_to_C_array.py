# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:42:52 2023

@author: mohan
"""


import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='Desktop/ML_Intern/converted_model.tflite')
interpreter.allocate_tensors()

# Get the model content as a byte array
model_content = interpreter.tensor(interpreter.get_input_details()[0]['index'])().tobytes()


# Save the model content as a C array
with open('Desktop/ML_Intern/model_data.h', 'w') as file:
    file.write("#include <stdint.h>\n\n")
    file.write("const uint8_t model_data[] = {")
    file.write(",".join([str(b) for b in model_content]))
    file.write("};\n")
