# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:58:01 2023

@author: jl2106
"""

from Kingdom import *
leave = "https://cdn.pixabay.com/photo/2018/03/25/10/02/farewell-3258939_1280.jpg"

# Class for this game
class Game_Kingdom():
    
    def __init__(self):
        self.kingdom = 1
   
    # Display for players
    def player_display(self, player, npc, npc2):
        clear_output()
        time.sleep(0.5)
        print("")
        print("Welcome to Kingdom - Chosun Dynasty !")
        time.sleep(0.5)
        player_name = input("Please write down your character's name: ") # Setting a character's name
        player.name = player_name
        time.sleep(0.5)
        ld_pick = 1
        casino_entrance_fee = random.randint(1, 5) # Setting the casino's entrence fee in this game
        while ld_pick == 1:
            print("Please select this game's level of difficulty") 
            time.sleep(0.5)
            print("0: Easy | 1: Normal | 2: Hard")
            try:
                a = int(input("The level of difficulty: ")) # Setting the level of difficulty
                time.sleep(0.5)
                if a == 0 :
                    player.hp = 200
                    player.natural_striking_power = 150
                    player.money = 50
                    ld_pick = 0
                    game_mode = player.game_mode
                elif a == 1:
                    player.hp = 150
                    player.natural_striking_power = 100
                    player.money = 30
                    ld_pick = 0
                    game_mode = player.game_mode
                elif a == 2:
                    player.hp = 100
                    player.natural_striking_power = 50
                    player.money = 10
                    ld_pick = 0
                    game_mode = player.game_mode
                else: print("Please put the right integer number!")
            except ValueError:
                print("Please put the right integer number!")

        # Print out the plot of this game             
        clear_output()
        time.sleep(1)
        print(f"{player.name} is supposed to get married to the princess of the Chosun Dynasty")
        time.sleep(1)
        print("However, everyone turned out to be a zombie, and we don't know the reason...")
        time.sleep(1)
        print("Therefore, you must save your princess!")
        time.sleep(1)
        print("It might be difficult, but you should try this for your princess")
        time.sleep(4)
        clear_output()
            
        while game_mode == 1:
            d = 0
            player.weight_check()
            player.striking_power_check()
            print("\n< Chosun Dynast Kingdom, save your princess! >") # Basic display 
            print(f"0. {player.name} details")
            print(f"1. {player.name} Backpack")
            print(f"2. Teleport to {npc.room}")
            print(f"3. Portal (in {npc.room})")
            print(f"4. Shopping (in {npc.room})")
            print(f"5. Selling (in {npc.room})")
            print(f"6. Deposit to Locker (in {npc.room})")
            print(f"7. Withdrawal from Locker (in {npc.room})")
            print(f"8. {player.name} Locker (in {npc.room})")
            print(f"9. Teleport to Casino (Entrance fee: £{casino_entrance_fee})")
            print("10. Give up this game")
            d = 1
            while d == 1:
                try:
                    a = input("Select a number, what you want to do?: ")
                    if type(int(a)) == int and int(a) < 11 :
                        a = int(a)
                        if a == 0: # Player Detail
                            print(f"{player.name}: HP({player.hp}/{player.hp_max}) | Striking power({player.striking_power})")
                            print(f"Current money(£{player.money}) | Weight({player.weight}/{player.weight_limit})")
                        elif a == 1: # Player Backpack
                            list_backpack = []
                            for i in player.backpack:
                                list_backpack.append(i.name)
                            print(f"Your backpack: {list_backpack}") 
                            print(f"Weight: {player.weight}/{player.weight_limit}")
                        elif a == 2: # Move to the NPC room
                            player.current_location = 1
                            print(f"Now You are in {npc.room}!")
                        elif a == 3: # Using the portal
                            if player.current_location == 1:
                                clear_output()
                                npc.portal(player)
                                d = 0
                            else: print(f"You are not in {npc.room} :(")
                        elif a == 4: # Shopping
                            if player.current_location == 1:
                                npc.shopping(player)
                                d = 0
                            else: print(f"You are not in {npc.room} :(")
                        elif a == 5: # Selling an item
                            if player.current_location == 1:
                                npc.selling(player)
                                d = 0
                            else: print(f"You are not in {npc.room} :(")
                        elif a == 6: # Deposit an item to the locker
                            if player.current_location == 1:
                                npc.deposit(player)
                                d = 0
                            else: print(f"You are not in {npc.room} :(")
                        elif a == 7: # Take out an item from the locker
                            if player.current_location == 1:
                                npc.withdrawal(player)
                                d = 0
                            else: print(f"You are not in {npc.room} :(")
                        elif a == 8: # Check the locker
                            if player.current_location == 1:
                                list_locker = []
                                for i in player.locker_list:
                                    list_locker.append(i.name)
                                    print(f"Your Locker: {list_locker}")
                            else: print(f"You are not in {npc.room} :(")
                        elif a == 9: # Enjoy the casino
                            if player.money - casino_entrance_fee > 0:
                                player.money -= casino_entrance_fee
                                player.current_location = 2
                                d = 0
                                npc2.casino(player)
                            else: print("You don't have enough moeny!")
                        elif a == 10: # Exit the game
                            print("Goodbye!")
                            time.sleep(1)
                            display(Image(url=leave, width=400, height=300))
                            player.game_mode = 0
                            game_mode = 0
                            d = 0
                    else:
                        print("Please put the right integer number!")  
                except ValueError:
                    print("Please put the right integer number!")
                
            if player.game_mode == 0:
                game_mode = 0
                break
            elif game_mode == 0:
                break     
            
    # Setting all elements of the game 
    def game_setting(self):
            
        # Create trophies
        self.trophy1 = Trophy("Broken Helmet", 3, 1, 5)
        self.trophy2 = Trophy("Broken Spear", 5, 1, 5)
        self.trophy3 = Trophy("Dirty Feather", 7, 1, 5)
        self.trophy4 = Trophy("Fragment of Shield", 9, 1, 5)
        self.trophy5 = Trophy("Gun Powder", 13, 1, 5)
        self.trophy6 = Trophy("Damaged Macbook Air M2", 5, 1, 5)
        self.trophy7 = Trophy("Damaged Macbook Pro M2 Max", 10, 1, 5)
        
        # Create keys
        self.training_key1 = Key("Training Key1", 1, 0.5, 1, 10)
        self.training_key2 = Key("Training Key2", 1, 0.5, 1, 11)
        self.boss_key = Key("King's bedroom key", 5000, 0.5, 1, 12)
        
        # Create NPCs
        self.sena = General_NPC("Sena")
        self.genesis = Casino_NPC("Genesis")
    
        # Creating and setting maps
        self.map1 = Normal_map("Main Entrance of Chosun Palace", 5, 3, 1, 10, 1)
        self.map1.setting_requirement()
        self.map1.add_trophy(self.trophy1)
        self.map1.story_add(["This is just a beginning...","So many zombies are here..."])
        self.sena.map_add(self.map1)
        self.map2 = Step_stage("Garden1: Princess' favorite place", 10, 3, 2, 20, 1)
        self.map2.setting_requirement()
        self.map2.add_trophy(self.trophy1)
        self.map2.add_key(self.training_key1)
        self.map2.story_add(["Why my princess likes this place?","Smells terrible here :D", "Anyway, I need more training"])
        self.sena.map_add(self.map2)
        self.map3 = Normal_map("Garden2: Queen's favorite place", 20, 3, 3, 10, 2)
        self.map3.setting_requirement()
        self.map3.story_add(["Oh my god, this is supposed to be our wedding venue","My princess' mother loved here :)"])
        self.map3.add_trophy(self.trophy2)
        self.sena.map_add(self.map3)
        self.map4 = Normal_map("Front Yard", 30, 3, 4, 15, 2)
        self.map4.setting_requirement()
        self.map4.add_trophy(self.trophy2)
        self.map4.story_add(["I am so tired...","But I cannot give up...", "Please wait, my princess!"])
        self.sena.map_add(self.map4)
        self.map5 = Normal_map("The only door of Main Building", 40, 3, 5, 10, 3)
        self.map5.setting_requirement()
        self.map5.add_trophy(self.trophy3)
        self.map5.story_add(["Am I ready to go inside?","It will be very tough..."])
        self.sena.map_add(self.map5)
        self.map6 = Step_stage("The First Section: Meeting Hall", 50, 3, 6, 15, 3)
        self.map6.setting_requirement()
        self.map6.add_trophy(self.trophy3)
        self.map6.add_key(self.training_key2)
        self.map6.story_add(["It wasn't easy...","I think I need more practice and better weapons!"])
        self.sena.map_add(self.map6)
        self.map7 = Normal_map("The Second Section: Kitchen", 60, 3, 7, 15, 4)
        self.map7.setting_requirement()
        self.map7.add_trophy(self.trophy4)
        self.map7.story_add(["Most of these foods are gone, it's so gross","My princess must be starving now..."])
        self.sena.map_add(self.map7)
        self.map8 = Normal_map("The Third Section: Library", 70, 3, 8, 20, 4)
        self.map8.setting_requirement()
        self.map8.add_trophy(self.trophy5)
        self.map8.story_add(["I used to read a lot of books at this library","However, I don't have time to read them now..."])
        self.sena.map_add(self.map8)
        self.map9 = Step_stage("The Last Section: Corridor", 90, 3, 9, 20, 5)
        self.map9.setting_requirement()
        self.map9.add_trophy(self.trophy5)
        self.map9.add_key(self.boss_key)
        self.map9.story_add(["They were my friends and comrades","Rest in peace, all of my friends...", "It's time to save my princess!"])
        self.sena.map_add(self.map9)
        self.map10 = Traning_map("Training Room 1", 30, 10, 10, 50)
        self.map10.add_trophy(self.trophy6)
        self.map10.key_setting(self.training_key1)        
        self.map10.map_setting(map_list = ["1F Corridor", "2F Corridor", "Chichester1 016", "Chichester1 017", "Chichester1 021", "Chichester1 023"],
                     monster_list = [("Undergraduate Student", 2),("Postgraduate Student",5),("Exchange Student",7),
                        ("TA",12),("Professor", 14)])
        self.map11 = Traning_map("Training Room 2", 60, 10, 11, 100)
        self.map11.add_trophy(self.trophy7)
        self.map11.key_setting(self.training_key2)
        self.map11.map_setting(map_list = ["1F Corridor", "2F Corridor", "Pevensey 012", "Pevensey 014", "Pevensey 021", "Pevensey 022"],
                     monster_list = [("Undergraduate Student", 6),("Postgraduate Student",10),("Exchange Student",14),
                        ("TA",20),("Professor", 24)])
        self.sena.map_add(self.map10)
        self.sena.map_add(self.map11)
        self.lee = Player("Jungyeon", 30000)
        self.boss_map = Boss_map("King's bedroom in Chosun Palace", "Victory", 5, 10, 1000)
        self.sena.map_add(self.boss_map)
        self.boss_map.key_setting(self.boss_key)
        self.boss_map.story_add(["I really missed you, my princess...","Let's go home, now you are safe!", "I will protect you forever...!"])

        # Create portions and weapons
        self.bandage = Portion("Bandage", 2, 0.5, 10, 30)
        self.ointment = Portion("Ointment", 5, 1, 10, 75)
        self.booster = Portion("Booster", 10, 1, 5, 150)
        self.dagger = Weapon("Dagger", 10, 4, 2, 25)
        self.s_sword = Weapon("Silver Sword", 20, 5, 1, 40)
        self.arrow = Weapon("Arrow", 25, 5, 1, 50)
        self.spear = Weapon("Spear", 35, 5, 1, 75)
        self.golden_axe = Weapon("Golden Axe", 50, 8, 1, 90)
        self.g_sword = Weapon("Golden Sword", 70, 8, 1, 100)
        self.epee = Weapon("Special épée", 100, 5, 1, 200)
        self.gun = Weapon("Revolver", 200, 10, 1, 300)
        self.bow = Weapon("Light Weight Bow", 250, 3, 1, 400)
        self.vibranium = Weapon("Vibranium Sword", 300, 15, 1, 500)

        # Put the items to the NPCs    
        self.sena.add_item(self.training_key1)
        self.sena.add_item(self.training_key2)
        self.sena.add_item(self.trophy1)
        self.sena.add_item(self.trophy2)
        self.sena.add_item(self.trophy3)
        self.sena.add_item(self.trophy4)
        self.sena.add_item(self.trophy5)
        self.sena.add_item(self.trophy6)
        self.sena.add_item(self.trophy7)
        self.sena.add_item(self.bandage)
        self.sena.add_item(self.ointment)
        self.sena.add_item(self.booster)   
        self.sena.add_item(self.dagger)
        self.sena.add_item(self.s_sword)
        self.sena.add_item(self.arrow)
        self.sena.add_item(self.spear)
        self.sena.add_item(self.golden_axe)
        self.sena.add_item(self.g_sword)
        self.sena.add_item(self.epee)
        self.sena.add_item(self.gun)
        self.sena.add_item(self.bow)
        self.sena.add_item(self.vibranium)
    
        self.sena.add_shop(self.bandage)
        self.sena.add_shop(self.ointment)
        self.sena.add_shop(self.booster)     
        self.sena.add_shop(self.dagger)
        self.sena.add_shop(self.s_sword)
        self.sena.add_shop(self.arrow)
        self.sena.add_shop(self.spear)
        self.sena.add_shop(self.golden_axe)
        self.sena.add_shop(self.g_sword)
        self.sena.add_shop(self.epee)
        self.sena.add_shop(self.gun)
        self.sena.add_shop(self.bow)
        self.sena.add_shop(self.vibranium)
    
        self.genesis.add_item(self.training_key1)
        self.genesis.add_item(self.training_key2)
        self.genesis.add_item(self.trophy1)
        self.genesis.add_item(self.trophy2)
        self.genesis.add_item(self.trophy3)
        self.genesis.add_item(self.trophy4)
        self.genesis.add_item(self.trophy5)
        self.genesis.add_item(self.trophy6)
        self.genesis.add_item(self.trophy7)
        self.genesis.add_item(self.dagger)
        self.genesis.add_item(self.s_sword)
        self.genesis.add_item(self.arrow)
        self.genesis.add_item(self.spear)
        self.genesis.add_item(self.golden_axe)
        self.genesis.add_item(self.g_sword)
        self.genesis.add_item(self.epee)
        self.genesis.add_item(self.gun)
        self.genesis.add_item(self.bow)
        self.genesis.add_item(self.vibranium)
        
        # Operate this game
        self.player_display(self.lee, self.sena, self.genesis)

        