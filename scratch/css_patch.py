with open('artifacts/selene-reg-x/src/index.css', 'r', encoding='utf-8') as f:
    content = f.read()

if '/* Custom Form Controls */' not in content:
    content += '''
/* Custom Form Controls */
select {
  -webkit-appearance: none;
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 0.5rem center;
  background-size: 1em;
  padding-right: 2rem !important;
}

option {
  background-color: var(--panel2) !important;
  color: var(--text) !important;
}

input[type="checkbox"],
input[type="radio"] {
  -webkit-appearance: none;
  appearance: none;
  background-color: var(--panel2);
  border: 1px solid var(--line);
  outline: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  cursor: pointer;
}

input[type="checkbox"] {
  border-radius: 4px;
  width: 14px;
  height: 14px;
}

input[type="radio"] {
  border-radius: 50%;
  width: 14px;
  height: 14px;
}

input[type="checkbox"]:checked,
input[type="radio"]:checked {
  background-color: var(--panel2);
  border-color: #f59e0b;
}

input[type="checkbox"]:checked::after {
  content: "";
  display: block;
  width: 8px;
  height: 8px;
  background-color: #f59e0b;
  border-radius: 1px;
}

input[type="radio"]:checked::after {
  content: "";
  display: block;
  width: 6px;
  height: 6px;
  background-color: #f59e0b;
  border-radius: 50%;
}
'''
    with open('artifacts/selene-reg-x/src/index.css', 'w', encoding='utf-8') as f:
        f.write(content)
print('Updated index.css')
