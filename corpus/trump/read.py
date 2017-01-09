import codecs

new = []

test = codecs.open('trump_speech_chatbot.txt', 'r')
out  = codecs.open('trump_long_speech.txt', 'w')

for i in test.readlines():
  if len(i.strip()) != 0:
    new.append(i)

for n in new:
  out.write(n)
