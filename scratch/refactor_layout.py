import re

filepath = "artifacts/selene-reg-x/src/App.tsx"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Extract the nav block from AppShell
nav_start = content.find('<nav aria-label="Primary navigation"')
nav_end_str = "</nav>"
nav_end = content.find(nav_end_str, nav_start) + len(nav_end_str)
nav_block = content[nav_start:nav_end]

# 2. Extract the nav array definition
array_start = content.find("const nav = [\n    ['workstation', '/', 'Workstation'")
array_end_str = "] as const;"
array_end = content.find(array_end_str, array_start) + len(array_end_str)
array_block = content[array_start:array_end]

# 3. Create the PrimaryNavigation component
primary_nav_component = f"""
export function PrimaryNavigation({{ current }}: {{ current: string }}) {{
  {array_block}

  return (
    {nav_block}
  );
}}
"""

# 4. Remove nav array and nav block from AppShell
new_content = content[:array_start] + content[array_end:]
new_nav_start = new_content.find('<nav aria-label="Primary navigation"')
new_nav_end = new_content.find(nav_end_str, new_nav_start) + len(nav_end_str)
new_content = new_content[:new_nav_start] + new_content[new_nav_end:]

# Insert PrimaryNavigation component before AppShell
appshell_start = new_content.find("function AppShell({")
new_content = new_content[:appshell_start] + primary_nav_component + "\n" + new_content[appshell_start:]

# 5. Insert PrimaryNavigation into Workstation after hero
hero_end_str = "</section>"
workstation_start = new_content.find("function Workstation() {")
hero_end = new_content.find(hero_end_str, workstation_start) + len(hero_end_str)

new_content = new_content[:hero_end] + "\n        <PrimaryNavigation current=\"workstation\" />\n" + new_content[hero_end:]

# 6. Insert PrimaryNavigation into Mosaic at top
mosaic_start = new_content.find("function Mosaic() {")
mosaic_return = new_content.find("<AppShell current=\"mosaic\"", mosaic_start)
mosaic_children_start = new_content.find(">", mosaic_return) + 1
new_content = new_content[:mosaic_children_start] + "\n      <PrimaryNavigation current=\"mosaic\" />" + new_content[mosaic_children_start:]

# 7. Insert PrimaryNavigation into Audit at top
audit_start = new_content.find("function Audit() {")
audit_return = new_content.find("<AppShell current=\"audit\"", audit_start)
audit_children_start = new_content.find(">", audit_return) + 1
new_content = new_content[:audit_children_start] + "\n      <PrimaryNavigation current=\"audit\" />" + new_content[audit_children_start:]

# 8. Insert PrimaryNavigation into Validation at top
val_start = new_content.find("function Validation() {")
val_return = new_content.find("<AppShell current=\"validation\"", val_start)
val_children_start = new_content.find(">", val_return) + 1
new_content = new_content[:val_children_start] + "\n      <PrimaryNavigation current=\"validation\" />" + new_content[val_children_start:]

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Refactored primary navigation layout.")
