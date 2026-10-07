import sys

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')

ws_start = content.find('function Workstation()')
ws_end = content.find('function Mosaic()')
code = content[ws_start:ws_end]
return_idx = code.rfind('  return (')

if return_idx == -1:
    print('return not found')
    sys.exit(1)

return_content = code[return_idx:return_idx+500]

# we just print it with backslash escaping to avoid unicode issues in powershell output
print(return_content.encode('unicode_escape').decode('utf-8'))
