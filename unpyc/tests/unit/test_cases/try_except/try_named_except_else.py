try:
    print('trying')
except ConnectionError as ce:
    print('exception')
else:
    print('no exception')
print('next thing')