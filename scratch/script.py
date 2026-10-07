import re
with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'<Gauge className=\"size-5[^\"]*\" />', '', content)
content = re.sub(r'<Target className=\"size-5[^\"]*\" />', '', content)
content = re.sub(r'<Activity className=\"size-5[^\"]*\" />', '', content)
content = re.sub(r'<FileImage className=\"size-4 text-sky-800\" />', '', content)
content = re.sub(r'<ScanSearch className=\"size-5[^\"]*\" />', '', content)

with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
