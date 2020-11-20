from PIL import Image, ImageOps
import numpy as np


# Activation func
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# convert image to binary array
def convertImg(img):
    grayscale_img = img.convert('L')
    grayscale_img = grayscale_img.resize((10, 10))  # gives 10x10 pixels img
    arr = np.array(grayscale_img)  # gives 10x10 array
    arr = np.reshape(arr, (100, 1))  # gives 100x1 array
    threshold = 250  # value for determining whether 1 or 0
    bin_arr = arr
    for i in range(100):
        if arr[i][0] > 250:
            bin_arr[i][0] = 1
        else:
            bin_arr[i][0] = 0
    return bin_arr


letter1_bin_arr = list()  # stores input arrays of first letter
for i in range(1, 6):
    letter1_bin_arr.append(convertImg(Image.open('s' + str(i) + '.png')))

letter2_bin_arr = list()  # stores input arrays of second letter
for i in range(1, 6):
    letter2_bin_arr.append(convertImg(Image.open('r' + str(i) + '.png')))

letter3_bin_arr = list()  # stores input arrays of third letter
for i in range(1, 6):
    letter3_bin_arr.append(convertImg(Image.open('a' + str(i) + '.png')))

training_outputs1 = np.array([[1, 0, 0]]).T  # correct o/p for letter1
training_outputs2 = np.array([[0, 1, 0]]).T  # correct o/p for letter2
training_outputs3 = np.array([[0, 0, 1]]).T  # correct o/p for letter3

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

for alpha in learning_rates:
    synaptic_weights1 = 2*np.random.random(
        (16, 100)) - 1  # gives a 16x100 array for weights btw first hidden layer and input layer
    synaptic_weights2 = 2*np.random.random(
        (3, 16)) - 1 # gives a 3x16 array for weights btw first hidden layer and output layer

    acceptable_error1 = False
    acceptable_error2 = False
    acceptable_error3 = False

    epochs = 0

    while True:

        for bin_arr in letter1_bin_arr:
            hidden_output = sigmoid(np.dot(synaptic_weights1, bin_arr))  # gives w(i)*x(i) = (16x100)*(100x1) = (16x1)
            output = sigmoid(np.dot(synaptic_weights2, hidden_output))  # gives w'(i)*z(i) = (3x16)*(16x1) = (3x1)

            error = training_outputs1 - output
            if (error[0] ** 2)/2 + (error[1] ** 2)/2 + (error[2] ** 2)/2 < 0.0001:
                acceptable_error1 = True
                break

            dely = (training_outputs1 - output) * sigmoid_derivative(output)
            delz = np.dot(synaptic_weights2.T, dely) * sigmoid_derivative(hidden_output)
            synaptic_weights2 += (alpha * np.dot(dely, hidden_output.T))
            synaptic_weights1 += (alpha * np.dot(delz, bin_arr.T))

        for bin_arr in letter2_bin_arr:
            hidden_output = sigmoid(np.dot(synaptic_weights1, bin_arr))  # gives w(i)*x(i) = (16x100)*(100x1) = (16x1)
            output = sigmoid(np.dot(synaptic_weights2, hidden_output))  # gives w'(i)*z(i) = (3x16)*(16x1) = (3x1)

            error = training_outputs2 - output
            if (error[0] ** 2)/2 + (error[1] ** 2)/2 + (error[2] ** 2)/2 < 0.0001:
                acceptable_error2 = True
                break

            dely = (training_outputs2 - output) * sigmoid_derivative(output)
            delz = np.dot(synaptic_weights2.T, dely) * sigmoid_derivative(hidden_output)
            synaptic_weights2 += (alpha * np.dot(dely, hidden_output.T))
            synaptic_weights1 += (alpha * np.dot(delz, bin_arr.T))

        for bin_arr in letter3_bin_arr:
            hidden_output = sigmoid(np.dot(synaptic_weights1, bin_arr))  # gives w(i)*x(i) = (16x100)*(100x1) = (16x1)
            output = sigmoid(np.dot(synaptic_weights2, hidden_output))  # gives w'(i)*z(i) = (3x16)*(16x1) = (3x1)

            error = training_outputs3 - output
            if (error[0] ** 2)/2 + (error[1] ** 2)/2 + (error[2] ** 2)/2 < 0.0001:
                acceptable_error3 = True
                break

            dely = (training_outputs3 - output) * sigmoid_derivative(output)
            delz = np.dot(synaptic_weights2.T, dely) * sigmoid_derivative(hidden_output)
            synaptic_weights2 += (alpha * np.dot(dely, hidden_output.T))
            synaptic_weights1 += (alpha * np.dot(delz, bin_arr.T))

        epochs += 1

        if acceptable_error1 and acceptable_error2 and acceptable_error3:
            break

     #testing sample to be inserted here
    img1 = Image.open('w1.png')
    greyscale_img1 = img1.convert('L')
    greyscale_img1 = greyscale_img1.resize((10,10))
    arr1 = np.array(greyscale_img1)
    arr1 = np.reshape(arr1 , (100,1))
    threshold = 250
    binary_arr1 = arr1 > threshold
    input_layer1 = binary_arr1
    outputs1 = sigmoid(np.dot( synaptic_weights1 , input_layer1 ))
    input_layer2 = outputs1
    outputs2 = sigmoid(np.dot( synaptic_weights2 , input_layer2 ))


    print("Outputs after training for learning rate : ")
    print(alpha)
    print(outputs2)
