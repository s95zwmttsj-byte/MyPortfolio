extends Node2D

var health
var nodeNum = 0

var stats = [0, 0, 0, 0, 0]
#key: vitality, strength, stealth, charisma, intelligence

var inventory = []

func load_json(file_path):
	var file = FileAccess.open(file_path, FileAccess.READ) 
	if file:
		var text = file.get_as_text()
		var json_result = JSON.parse_string(text)
		return json_result
	
func requirement_display(requirements):
	var text = "Requirements: "
	for i in requirements.size():
		if str(requirements[i]) != "0" and str(requirements[i]) != "":
			match i:
				0:
					text += "Vitality: " + str(requirements[i]) + ", "
				1: 
					text += "Strength: " + str(requirements[i]) + ", "
				2: 
					text += "Stealth: " + str(requirements[i]) + ", "
				3:
					text += "Charisma: " + str(requirements[i]) + ", "
				4: 
					text += "Intelligence: " + str(requirements[i]) + ", "
				5: 
					text += "Item: " + requirements[i] + " "
	return text

func text_constructor(node):
	var file_path =  "res://story_data.json"
	var story_data = load_json(file_path)
	var text = ""
	text += story_data[node]["story"] + "\n \n"
	for i in story_data[node]["choices"].size():
		text += str(i + 1) + ". "
		text += story_data[node]["choices"][i]["text"] + "\n"
		text += requirement_display(story_data[node]["choices"][i]["requirements"])
		text += "\n \n"
	return text

func _process(delta):
	$Vitality.text = str(stats[0])
	$Strength.text = str(stats[1])
	$Stealth.text = str(stats[2])
	$Charisma.text = str(stats[3])
	$Intelligence.text = str(stats[4])
	$Dialogue.text = text_constructor(nodeNum)




func _on_button_pressed():
	var file_path =  "res://story_data.json"
	var story_data = load_json(file_path)
	nodeNum = story_data[nodeNum]["choices"][0]["next"] - 1




func _on_option_2_pressed():
	var file_path =  "res://story_data.json"
	var story_data = load_json(file_path)
	nodeNum = story_data[nodeNum]["choices"][1]["next"] - 1
