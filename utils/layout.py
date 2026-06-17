def box(message: str):
    max_length = max(len(line) for line in message.split('\n'))
    print('+' + '-' * (max_length + 4) + '+')
    for i in message.split('\n'):
        print(f"|  {i}" + ' ' * (max_length - len(i)) + '  |')
    print('+' + '-' * (max_length + 4) + '+')
