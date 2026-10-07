import re

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')
    
idx = content.find('function Workstation()')
sub_content = content[idx:idx+3000]

# Print out where the return statement starts.
match = re.search(r'return\s*\(\s*<AppShell[^>]*>', sub_content)
if match:
    print(repr(sub_content[match.start():match.end() + 200]))
else:
    print('Not found')
