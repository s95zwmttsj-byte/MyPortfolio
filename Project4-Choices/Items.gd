extends RichTextLabel

func _inventory():
	var returnText = ""
	for i in $"..".inventory.size():
		returnText += str($"..".inventory[i]) + "\n"
	return returnText

func _process(delta):
	text = "Inventory: \n \n" + _inventory()
