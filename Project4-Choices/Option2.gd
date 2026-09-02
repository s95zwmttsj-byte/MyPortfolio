extends Button

func load_json(file_path):
	var file = FileAccess.open(file_path, FileAccess.READ) 
	if file:
		var text = file.get_as_text()
		var json_result = JSON.parse_string(text)
		return json_result

func requirements_met():
	var met = true
	var file_path =  "res://story_data.json"
	var story_data = load_json(file_path)
	var requirements = story_data[$"..".nodeNum]["choices"][1]["requirements"]
	var item_requirement = requirements.pop_back()
	for i in requirements.size():
		if $"..".stats[i] < requirements[i]:
			met = false
	if item_requirement == "":
		return met
	else:
		for i in $"..".inventory.size():
			if $"..".inventory[i] == item_requirement:
				return met
		met = false
		return met
		
	
	
func _process(delta):
	if !requirements_met():
		disabled = true
	else:
		disabled = false


func _on_pressed():
	var file_path =  "res://story_data.json"
	var story_data = load_json(file_path)
	if $"..".nodeNum == 0:
		$"..".inventory = []
		$"..".stats = [0,0,0,0,0]
	if story_data[$"..".nodeNum]["gaineditems"] != "":
		$"..".inventory.append(story_data[$"..".nodeNum]["gaineditems"])
	for i in $"..".stats.size():
		$"..".stats[i] += story_data[$"..".nodeNum]["gainedstats"][i]
	if story_data[$"..".nodeNum]["lostitems"] != "":
		$"..".inventory.erase(story_data[$"..".nodeNum]["lostitems"])
