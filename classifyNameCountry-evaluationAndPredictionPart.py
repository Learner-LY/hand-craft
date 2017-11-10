# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output
# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomChoice()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = category_tensor.data[0]
    confusion[category_i][guess_i] += 1
# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()
# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)
# Set up axes
ax.set_xticklabels([''] + dic.keys(), rotation=90)
ax.set_yticklabels([''] + dic.keys())
# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# sphinx_gallery_thumbnail_number = 2
plt.show()

def predict(user_input,num):
    input = Variable(wordToTensor(user_input))
    hidden = Variable(torch.zeros(1, n_hidden))
    for i in range(input.size()[0]):
        output,next_hidden=rnn(input[i],hidden)
    top_n,top_i=output.data.topk(num)
    re=[]
    for i in range(num):
        re.append(top_i[0][i])   #因为top_i是1*num的;
        print dic.keys()[re[i]]
    return re
