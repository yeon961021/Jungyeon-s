# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:54:05 2023

@author: jl2106
"""
import random
import numpy as np
from IPython.display import clear_output, display, Image
win = "https://cdn.pixabay.com/photo/2019/06/10/18/46/welsh-corgi-pembroke-4264958_1280.jpg"
defeat = "https://cdn.pixabay.com/photo/2017/12/29/20/44/suppression-3048645_1280.jpg"
import time

# Map class
class Map:
    
    # Basic settings of a map
    def __init__(self, name, reward, attempts, map_number):
        self.name = name
        self.reward = reward
        self.attempts = attempts
        self.map_number = map_number
        self.requirement = 0
        self.open = 0
        self.game_on = 0
        self.monster_name = ""
        
    # Setting a requirement based on the monster type and the number of monsters  
    def setting_requirement(self):
        if self.monster_type == 1:
            monster_strength = 5
            self.monster_name = "Gate Keeper"
            self.requirement = self.monster_number * monster_strength
        elif self.monster_type == 2:
            monster_strength = 10
            self.monster_name = "Spearman"
            self.requirement = self.monster_number * monster_strength
        elif self.monster_type == 3:
            monster_strength = 20
            self.monster_name = "Archer"
            self.requirement = self.monster_number * monster_strength
        elif self.monster_type == 4:
            monster_strength = 30
            self.monster_name = "Warrior"
            self.requirement = self.monster_number * monster_strength
        elif self.monster_type == 5:
            monster_strength = 35
            self.monster_name = "Special Soldier"
            self.requirement = self.monster_number * monster_strength
        else:
            return False
        
    # Adding stories for this map    
    def story_add(self, story_list):
        self.story = story_list
        
    # Printing the stroies when players finish this map    
    def print_story(self):
        print("")
        for story in self.story:
            time.sleep(1.5)
            print(f"|  {story}  |")
        while True:
            try:
                time.sleep(1.5)
                decision = int(input("Exit = -1: "))
                time.sleep(0.5)
                if decision == -1:
                    break
                else: print("Please put the right integer number!")
            except ValueError:
                print("Please put the right integer number!")  

    # If the player wins this map, provide a reward and finish this map        
    def player_win(self, player):
        time.sleep(1)
        player.money += self.reward
        player.map_count += 1
        self.attempts = 0
        print(f"You just cleared this map, you have got total £{self.reward}! : HP: {player.hp}| Current money: £{player.money}")
        time.sleep(1)
        self.lucky_draw_money(player) # Conducting a lucky draw for money
        time.sleep(1)
        self.lucky_draw_trophy(player) # Conducting a lucky draw for a trophy
        self.print_story() # Call the 'print_stroy()' function
        self.game_on = 0 

    # If the player loses this map, finish this map with a health penalty
    def player_lose(self, player):
        player.hp = 10
        print("You failed to complete this map, train hard!")
        time.sleep(1)
        self.open = 0
        display(Image(url=defeat, width=400, height=300))
        self.game_on = 0

    # Basic information of this map's monster 
    def print_detail(self):
        print(f"[You must fight with {self.monster_number} {self.monster_name}s]")
        time.sleep(0.5)
        print("[You can choose one of the attack options]")
 
    # Lucky draw for money    
    def lucky_draw_money(self, player):
        draft_draw_list = list(range(1,100))
        random.shuffle(draft_draw_list)
        i = random.randint(0,98)
        i2 = random.randint(1,99)
        if draft_draw_list[i] == i2:
            time.sleep(2.0)
            print("£££££££££££££££££££££££££££££££££££££££££££££££££££££££")
            print(f"You are so lucky, you just got £{self.reward*7} more!")
            print("£££££££££££££££££££££££££££££££££££££££££££££££££££££££")
            player.money += (self.reward * 7)
            time.sleep(2.0)

    # Setting a key item when needed        
    def key_setting(self, key):
        self.key = key

    # Checking that a player has the key when needed
    def check_key(self, player):
        for item in player.backpack:
            if item.name == self.key.name:
                self.open = 1
                return True              

    # Setting a trophy for this map   
    def add_trophy(self, trophy):
        self.trophy = trophy     

    # Lucky draw for trophy  
    def lucky_draw_trophy(self, player):
        i = random.randint(1,4)
        if i == 2:
            print(f"Lucky! you can pick up a {self.trophy.name}!")
            time.sleep(0.5)
            while i != 0:
                try:
                    a = int(input("If you want to pick it up, put 1 [Exit: -1]: "))
                    time.sleep(0.5)
                    if a == 1:
                        if player.weight_check() <= (player.weight_limit - self.trophy.weight):
                            player.backpack.append(self.trophy)
                            print(f"You got the {self.trophy.name}!")
                            i = 0
                        else:
                            while True:
                                try:
                                    choose = int(input("Your backpack is full! [Drop your items : 1 | Exit: 2]: "))
                                    time.sleep(0.5)
                                    if choose == 1: 
                                        player.backpack_drop()
                                        break
                                    elif choose == 2:
                                        i = 0
                                        break
                                    else: print("Please put the right integer number!")
                                except ValueError:
                                    print("Please put the right integer number!")
                    elif a == -1:
                        i = 0
                    else: print("Please put the right integer number!")
                except ValueError:
                    print("Please put the right integer number!")

    # Checking the player's Health Points to determine the result of the battle
    def hp_check(self, player):
        if player.hp > 0:
            return True
        else:
            return False                    

    # Starting to explore a map              
    def start_game(self, player, npc):
        self.attempts -= 1
        print(f"[{self.name}]")
        time.sleep(1)
        print(f"You are fighting with {self.monster_number} {self.monster_name}s!")
        time.sleep(1)
        if player.striking_power >= self.requirement:
            damage_from_monster = random.randint(1, self.requirement*0.7)
            player.hp -= damage_from_monster
            if self.hp_check(player) == True: # Call the 'hp_check()' function to check the player's HP
                self.player_win(player) # Call the 'player_win()' function
            else:
                self.player_lose(player) # Call the 'player_lose()' function
        else: 
            self.player_lose(player) # Call the 'player_lose()' function

# Storyline Map Class (Normal) (from Map) 
class Normal_map(Map):
    
    def __init__(self, name, reward, attempts, map_number, monster_number, monster_type):
        super().__init__(name, reward, attempts, map_number)
        self.monster_number = monster_number
        self.monster_type = monster_type
        self.requirement = 0
        self.map_type = 1
        self.monster_name = ""
        
# Storyline Map Class (Key map) (from Map) 
class Step_stage(Map):
    
    def __init__(self, name, reward, attempts, map_number, monster_number, monster_type):
        super().__init__(name, reward, attempts, map_number)
        self.requirement = 0
        self.map_type = 1
        self.monster_number = monster_number
        self.monster_type = monster_type
        self.monster_name = ""
        self.game_on = 0
    
    # Adding a key for the player to enter a certain map
    def add_key(self, key):
        self.key = key

    # Starting to explore a map          
    def start_game(self, player, npc):
        self.attempts -= 1
        lucky_damage = 0
        print("")
        print(f"Welcome to {self.name}, you have to navigate this section from now on")
        self.game_on = 1
        time.sleep(0.5)
        print(f"[You must fight with {self.monster_number} {self.monster_name}s]")
        time.sleep(0.5)
        print("[You can choose one of the attack options]")
        time.sleep(0.5)
        while self.game_on == 1:
            try:
                a = input("1. Full Attack (put: FA) | 2. Lucky Attack (put: LA): ").lower()
                time.sleep(1)
                if a == 'fa': # Full Attack section with key drop
                    if player.striking_power >= self.requirement:
                        damage_from_monster1 = random.randint(1, self.requirement*0.5)
                        player.hp -= damage_from_monster1 
                        if self.hp_check(player) == True:  
                            self.player_win(player)
                            player.add_item(self.key) # Key drop
                            map_name = ""
                            for i in npc.map_list:
                                if i.map_number == self.key.game_map:
                                    map_name = i.name
                            time.sleep(0.5)
                            print(f"Now, you have a {self.key.name}, you can enter {map_name}!")
                            time.sleep(0.5)
                            break
                        else:
                            self.player_lose(player)
                            break
                    else:
                        self.player_lose(player)
                        break
                    
                elif a == 'la': # Lucky Attack section with key drop
                    lucky_damage = round(player.striking_power * round(np.random.uniform(0.5, 1.5),1)) # Calculating the lucky damage
                    if lucky_damage >= self.requirement:
                        damage_from_monster1 = random.randint(1, self.requirement*0.5) # Damage from the monster
                        player.hp -= damage_from_monster1  
                        if self.hp_check(player) == True:  
                            if player.striking_power >= lucky_damage:
                                loss = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                player.lucky_striking_power -= int(loss) # Replace the original striking power with the new striking power from the lucky attack
                                player.striking_power_check()
                                print(f"You just lost {loss} SP | Current: {player.striking_power} SP")
                                self.player_win(player)
                            else:
                                gross = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                player.lucky_striking_power += int(gross) # Replace the original striking power with the new striking power from the lucky attack
                                player.striking_power_check()
                                print(f"You just got {gross} SP | Current: {player.striking_power} SP")
                                self.player_win(player)
                            player.add_item(self.key) # Key drop
                            map_name = ""
                            for i in npc.map_list:
                                if i.map_number == self.key.game_map:
                                    map_name = i.name
                            time.sleep(0.5)
                            print(f"Now, you have a {self.key.name}, you can enter {map_name}!")
                            time.sleep(0.5)

                            break
                        else:
                            self.player_lose(player)
                            break
                        
                    else:
                        self.player_lose(player)
                        break
                    
                else: 
                    print("Please put the right number!")
                    
            except ValueError:
                print("Please put the right integer number!")       

# Training Map Class (from Map) 
class Traning_map(Map):
    def __init__(self, name, reward, attempts, map_number, requirement):
        super().__init__(name, reward, attempts, map_number)
        self.requirement = requirement
        self.section_requirement = 0
        self.open = 0
        self.map_type = 1
        self.monster_name = ""
        self.monster_number = 0
        self.current_location = ""
        self.map_name_list = []
        self.instuction_1 = 0
        self.instuction_2 = 0
        self.instuction_3 = 0
        self.instuction_4 = 0
        self.instuction_5 = 0
        self.instuction_6 = 0
        self.section = 0
        self.game_on = 0
        self.win_count = 0
        self.fight_on = 0
        self.fight_option = ""
        self.attack_option = 0

    # Setting a map from the map and monster lists
    def map_setting(self, map_list, monster_list):
        self.map_name_list = map_list
        self.monster_list = monster_list
        # Setting the sections' name
        self.section_1 = map_list[0]
        self.section_2 = map_list[1]
        self.section_3 = map_list[2]
        self.section_4 = map_list[3]
        self.section_5 = map_list[4]
        self.section_6 = map_list[5]
        self.map_number = 0.5 # For the 'player.map_count()' function

    # Lucky draw for money at each section
    def lucky_draw_money(self, player):
        draft_draw_list = list(range(1,100))
        random.shuffle(draft_draw_list)
        section_reward = round(self.reward/6)
        i = random.randint(0,98)
        i2 = random.randint(1,99)
        if draft_draw_list[i] == i2:
            time.sleep(2.0)
            print("£££££££££££££££££££££££££££££££££££££££££££££££££££££££")
            print(f"You are so lucky, you just got £{self.reward*7} more!")
            print("£££££££££££££££££££££££££££££££££££££££££££££££££££££££")
            player.money += (section_reward * 7)
            time.sleep(2.0)

    # Finish this map when the player clears all the sections        
    def player_win(self, player):
        print(f"You just cleared this map! | HP: {player.hp} | Current money: £{player.money}")
        time.sleep(0.5)
        self.game_on = 0 

    # Print the instructions when players arrive in each section    
    def print_instructions(self, player):
        self.location_check(player)
        time.sleep(0.5)
        print("")
        print(f"[You are now in {self.current_location}]")
        time.sleep(0.5)
        if self.win_check(player) == True:
            self.player_win(player)
        else:
            # Print the instruction first and make them fight with the monsters
            if self.section == 1 and self.instuction_1 == 0:
                self.print_detail()
                self.instuction_1 = 1
                self.fight(player) # Call the 'fight()' function
            elif self.section == 2 and self.instuction_2 == 0:
                self.print_detail()
                self.instuction_2 = 1
                self.fight(player)
            elif self.section == 3 and self.instuction_3 == 0:
                self.print_detail()
                self.instuction_3 = 1
                self.fight(player)
            elif self.section == 4 and self.instuction_4 == 0:
                self.print_detail()
                self.instuction_4 = 1
                self.fight(player)
            elif self.section == 5 and self.instuction_5 == 0:
                self.print_detail()
                self.instuction_5 = 1
                self.fight(player)
            elif self.section == 6 and self.instuction_6 == 0:
                self.print_detail()
                self.instuction_6 = 1
                self.fight(player)
            else: # When the player already cleared this section
                print("You already cleared this section!")
                self.section_select(player)

    # Checking the current location of the player and setting the details of the monster
    def location_check(self, player):
        monster, monster_strength = random.choice(self.monster_list)
        self.monster_name = monster
        self.monster_number = round(random.uniform(5, 12))
        self.section_requirement = round((monster_strength * self.monster_number)*0.5)
        self.current_location = self.map_name_list[self.section-1]

    # Enable the player to navigate this map       
    def section_select(self, player):
        while self.game_on == 1:
            select_mode = 1
            print("Which direction do you want to go?")
            time.sleep(0.5)
            while select_mode == 1:
                if self.section == 1: # 1st Section
                    selected_section = input(f"1. {self.section_3} | 2. {self.section_2} : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 3
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 2
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")
                elif self.section == 2: # 2nd Section
                    selected_section = input(f"1. {self.section_1} | 2. {self.section_5}  : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 1
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 5
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")
                elif self.section == 3: # 3rd Section
                    selected_section = input(f"1. {self.section_1} | 2. {self.section_4} : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 1
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 4
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")
                elif self.section == 4:  # 4th Section
                    selected_section = input(f"1. {self.section_3} : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 3
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 4
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")   
                elif self.section == 5: # 5th Section
                    selected_section = input(f"1. {self.section_2} | 2. {self.section_6} : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 2
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 6
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")   
                elif self.section == 6: # 6th Section
                    selected_section = input(f"1. {self.section_5} : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 5
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) != 1: 
                        print("Please put the right number!")   
                    else: print("Please put the right number!")

   # Checking whether the player has cleared all sections     
    def win_check(self, player):
        if self.win_count == 6:
            return True
        else: 
            return False

    # Battle with the zombies using two attack methods (with section rewards)        
    def fight(self, player):
        fight_on = 1
        section_reward = round(self.reward/6)
        while fight_on == 1:
            try:
                attack_option = input("1. Full Attack (Put: FA) | 2. Lucky Attack (Put: LA): ").lower()
                if attack_option == 'fa': # Full Attack
                    time.sleep(0.5)
                    fight_on = 0
                    if player.striking_power >= self.section_requirement:
                        damage_from_monster = round(random.uniform(1, self.section_requirement*0.7)) 
                        player.hp -= damage_from_monster
                        self.win_count += 1  # When the player clears one section, this win_count increases by one point
                        if self.hp_check(player) == True and self.win_check(player) == False:
                            player.money += section_reward
                            print(f"You just cleared {self.current_location}!, you just earned £{section_reward}!")
                            print(f"You current status: HP: {player.hp} | Current money: £{player.money}")
                            self.lucky_draw_money(player)
                            self.lucky_draw_trophy(player)
                            self.section_select(player)
                        elif self.hp_check(player) == True and self.win_check(player) == True:
                            player.money += section_reward
                            print(f"You just cleared {self.current_location}!, you just earned £{section_reward}!")
                            self.player_win(player)
                        elif self.hp_check(player) == False:
                            self.player_lose(player)
                    else:
                        self.player_lose(player)
                elif attack_option == 'la': # Lucky Attack
                    lucky_damage = 0
                    lucky_damage = round(player.striking_power * round(np.random.uniform(0.5, 1.5),1))
                    fight_on = 0
                    if lucky_damage >= self.section_requirement:
                        damage_from_monster = round(random.uniform(1, self.section_requirement*0.7))
                        player.hp -= damage_from_monster
                        self.win_count += 1  # When the player clears one section, this win_count increases by one point
                        if self.hp_check(player) == True:
                            if player.striking_power >= lucky_damage and self.win_check(player) == False:
                                loss = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} but you just lost {loss} SP from Lucky Attack")
                                player.lucky_striking_power -= int(loss)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                player.money += section_reward
                                print(f"Section Reward: £{section_reward} | Current money: £{player.money}")
                                self.lucky_draw_money(player)
                                time.sleep(0.5)
                                self.lucky_draw_trophy(player)
                                self.section_select(player)
                            elif player.striking_power < lucky_damage and self.win_check(player) == False:
                                gross = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} and you also got {gross} strike power!")
                                player.lucky_striking_power += int(gross)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                player.money += section_reward
                                print(f"Section Reward: £{section_reward} | Current money: £{player.money}") 
                                self.lucky_draw_money(player)
                                time.sleep(0.5)
                                self.lucky_draw_trophy(player)
                                self.section_select(player)
                            elif player.striking_power >= lucky_damage and self.win_check(player) == True:
                                loss = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} but you just lost {loss} SP from Lucky Attack")
                                player.lucky_striking_power -= int(loss)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                player.money += section_reward
                                print(f"Section Reward: £{section_reward} | Current money: £{player.money}") 
                                self.player_win(player)
                            elif player.striking_power < lucky_damage and self.win_check(player) == True:
                                gross = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} and you also got {gross} strike power!")
                                player.lucky_striking_power += int(gross)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                player.money += section_reward
                                print(f"Section Reward: £{section_reward} | Current money: £{player.money}")  
                                self.player_win(player)
                        elif self.hp_check(player) == False:
                            self.player_lose(player)                  
                    else:
                        self.player_lose(player)               
                else:
                    print("Invalid option. Please choose FA or LA")
            except Exception as e: # Debugging any errors from this function
                print(f"An error occurred: {e}")
 
    # Starting to explore a training map     
    def start_game(self, player, npc):
        if self.check_key(player) == True: # Check whether the player has a key
            self.attempts -= 1
            self.instuction_1 = 0
            self.instuction_2 = 0
            self.instuction_3 = 0
            self.instuction_4 = 0
            self.instuction_5 = 0
            self.instuction_6 = 0
            self.game_on = 1
            self.win_count = 0
            print("")
            time.sleep(1)
            print(f"Welcome to {self.name}, you have to navigate this map from now on")
            time.sleep(1)
            self.section = 1 # Setting the player's first section
            while self.game_on == 1:
                # Based on the 'print_instructions()' function, this map will be operated       
                self.print_instructions(player)
                # Logic: print_instructions() -> print_detail() -> fight() -> hp_check() and win_check() -> section_select() -> print_instructions() and win_check()
        else:
            print(f"You don't have the key, you cannot enter {self.name}")

# Boss Map Class (Final Stage) (from Map)       
class Boss_map(Map):
    def __init__(self, name, reward, attempts, map_number, requirement):
        super().__init__(name, reward, attempts, map_number)
        self.name = name
        self.monster_type = "Boss"
        self.requirement = requirement
        self.map_type = 2
        self.attempts = 5
        self.reward = "Victory"
        self.map_number = map_number
        self.open = 0
        self.monster_name = "" 
        self.current_location = ""
        self.instuction_1 = 0
        self.instuction_2 = 0
        self.instuction_3 = 0
        self.instuction_4 = 0
        self.section = 0
        self.game_on = 0

    # Basic information of the boss map's monster
    def print_detail(self):
        print(f"[You must fight with a {self.monster_name}]")
        time.sleep(0.5)
        print("[You can choose one of the attack options]")  

    # Print the instructions when players arrive in each section (Boss map)              
    def print_instructions(self, player):
        self.location_check(player)
        print("")
        print(f"[You are now in {self.current_location}]")
        time.sleep(0.5)
        if self.section == 1 and self.instuction_1 == 0:
            self.print_detail()
            self.instuction_1 = 1
            self.fight(player)
        elif self.section == 2 and self.instuction_2 == 0:
            self.print_detail()
            self.instuction_2 = 1
            self.fight(player)
        elif self.section == 3 and self.instuction_3 == 0:
            self.print_detail()
            self.instuction_3 = 1
            self.fight(player)
        elif self.section == 4 and self.instuction_4 == 0:
            self.print_detail()
            self.instuction_4 = 1
            self.fight(player)
        else: 
            print("You already cleared this map!")
            self.section_select(player)

    # Checking the current location of the player and setting the details of the monster (Boss map)        
    def location_check(self, player):
        if self.section == 1:
            self.current_location = "1F Corridor"
            self.monster_name = "creepy eunuch" 
        elif self.section == 2:
            self.current_location = "Room Number 2"
            self.monster_name = "contaminated prince" 
        elif self.section == 3:
            self.current_location = "2F Corridor"
            self.monster_name = "king's best body guard" 
        elif self.section == 4:
            self.current_location = "King's Room"
            self.monster_name = "ZOMBIE KING" 

    # Enable the player to navigate the boss map      
    def section_select(self, player):
        while self.game_on == 1:
            select_mode = 1
            print("Which direction do you want to go?")
            while select_mode == 1:
                if self.section == 1: # Corridor
                    selected_section = input("1. Upstairs | 2. Room Number 2 : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 3
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 2
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")
                elif self.section == 2: # Room Number 2
                    selected_section = input("1. Back to the corridor : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 1
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) != 1: 
                        print("Please put the right number!")   
                    else: print("Please put the right number!")
                elif self.section == 3: # Upstairs
                    selected_section = input("1. King's Room | 2. 1F floor : ")
                    if selected_section.isnumeric() and int(selected_section) == 1:
                        self.section = 4
                        select_mode = 0
                        self.print_instructions(player)
                    elif selected_section.isnumeric() and int(selected_section) == 2:
                        self.section = 1
                        select_mode = 0
                        self.print_instructions(player)
                    else: print("Please put the right number!")
                    
    # Checking whether the player won the battle with the boss monster (Zombie King)
    def win_check(self, player):
        if self.instuction_4 == 1: # Boss section
            return True
        else: 
            return False

    # Print the reward if the player successfully finishes the boss map, and turn off the whole game        
    def player_win(self, player):
        time.sleep(1)
        self.print_story()
        time.sleep(1)
        print("Game finished, you win!")
        time.sleep(1)
        display(Image(url=win, width=400, height=300))
        player.game_mode = 0
        self.game_on = 0

    # Battle with the zombies using two attack methods (Boss map)        
    def fight(self, player):
        fight_on = 1
        while fight_on == 1:
            try:
                attack_option = input("1. Full Attack (put: FA) | 2. Lucky Attack (put: LA): ").lower()
                if attack_option == 'fa':
                    time.sleep(0.5)
                    fight_on = 0
                    if player.striking_power >= self.requirement:
                        damage_from_monster = round(random.uniform(1, self.requirement*0.5)) 
                        player.hp -= damage_from_monster
                        if self.hp_check(player) == True and self.win_check(player) == False:
                            print(f"You just cleared {self.current_location}! | Current HP: {player.hp}")
                            self.section_select(player)
                        elif self.hp_check(player) == True and self.win_check(player) == True:
                            self.player_win(player)
                        elif self.hp_check(player) == False:
                            self.player_lose(player)
                    else:
                        self.player_lose(player)
                elif attack_option == 'la':
                    lucky_damage = 0
                    lucky_damage = round(player.striking_power * round(np.random.uniform(0.5, 1.5),1))
                    fight_on = 0
                    if lucky_damage >= self.requirement:
                        damage_from_monster = round(random.uniform(1, self.requirement*0.5))
                        player.hp -= damage_from_monster
                        if self.hp_check(player) == True:
                            if player.striking_power >= lucky_damage and self.win_check(player) == False:
                                loss = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} but you just lost {loss} SP from Lucky Attack")
                                player.lucky_striking_power -= int(loss)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                self.section_select(player)
                            elif player.striking_power < lucky_damage and self.win_check(player) == False:
                                gross = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} and you also got {gross} strike power!")
                                player.lucky_striking_power += int(gross)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                self.section_select(player)
                            elif player.striking_power >= lucky_damage and self.win_check(player) == True:
                                loss = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} but you just lost {loss} SP from Lucky Attack")
                                player.lucky_striking_power -= int(loss)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                self.player_win(player)
                            elif player.striking_power < lucky_damage and self.win_check(player) == True:
                                gross = abs(player.striking_power - lucky_damage)
                                time.sleep(0.5)
                                print(f"You just cleared {self.current_location} and you also got {gross} strike power!")
                                player.lucky_striking_power += int(gross)
                                player.striking_power_check()
                                print(f"You current status: HP: {player.hp} | Striking Power : {player.striking_power}")
                                time.sleep(0.5)
                                self.player_win(player)
                        elif self.hp_check(player) == False:
                            self.player_lose(player)                   
                    else:
                        self.player_lose(player)               
                else:
                    print("Invalid option. Please choose FA or LA")
            except Exception as e:
                print(f"An error occurred: {e}")
                
    # Starting to explore the boss map                 
    def start_game(self, player, npc):
        self.check_key(player)
        if self.check_key(player) == True:
            self.attempts -= 1
            self.instuction_1 = 0
            self.instuction_2 = 0
            self.instuction_3 = 0
            self.instuction_4 = 0
            self.game_on = 1
            print("")
            time.sleep(1)
            print(f"Welcome to {self.name}, you have to navigate this map from now on")
            time.sleep(1)
            self.section = 1
            self.print_instructions(player)
        else:
            print(f"You don't have a key, you cannot enter {self.name}")
        
# Item Class        
class Item:
    
    # The basic parameters of all items
    def __init__(self, name, price, weight, limit):
        self.name = name
        self.price = price
        self.weight = weight
        self.limit = limit

    # Append an item to the player's backpack        
    def add_item(self, player, item):
        if player.wegiht < 30 - item.weight: # Check the weight because the player's backpack has a weight limit
            player.backpack.append(item)
            print(f'You just got a {item.name}')
        else: print('Your backpack is full!')

# Key Class (from Item)     
class Key(Item):
    
    def __init__(self, name, price, weight, limit, game_map):
        super().__init__(name, price, weight, limit)
        self.game_map = game_map
        self.power = 0
        
    def add_item(self, player, item): # A key can be appenned to the player's backpack even if the player doesn't have enough space
        player.backpack.append(item)

# Portion Class (from Item)           
class Portion(Item):
    
    def __init__(self, name, price, weight, limit, hp_up):
        super().__init__(name, price, weight, limit)
        self.hp_up = hp_up
        self.power = 0
        
    def detail(self): # Details for the 'shopping()' function
        print(f'{self.name}: Price :£{self.price} | Weight: {self.weight}kg | Limit: {self.limit} pcs | + {self.hp_up}HP')

# Weapon Class (from Item)             
class Weapon(Item):
    #Weapon(name, price, weight, limit, power)
    def __init__(self, name, price, weight, limit, power):
        super().__init__(name, price, weight, limit)
        self.power = power
        
    def detail(self): # Details for the 'shopping()' function
        print(f'{self.name}: Price :£{self.price} | Weight: {self.weight}kg | Limit: {self.limit} pcs | + {self.power} striking power')

# Trophy Class (from Item)         
class Trophy(Item):
    #Trophy(name, price, weight, limit, monster)
    def __init__(self, name, price, weight, limit):
        super().__init__(name, price, weight, limit)
        self.power = 0  

# Player Class        
class Player:
    
    # Basic settings of a player
    def __init__(self, name, hp):
        self.name = name
        self.hp = hp
        self.hp_max = 5000
        self.locker_list = []
        self.backpack = []
        self.natural_striking_power = 0
        self.item_striking_power = 0 # Striking power from the items
        self.lucky_striking_power = 0 # Striking power from the lukcy attack
        self.striking_power = self.natural_striking_power + self.item_striking_power
        self.money = 0
        self.weight = 0
        self.weight_limit = 40.0
        self.current_location = 0
        self.game_mode = 1
        self.map_count = 1

    # Append an item to the player's backpack
    def add_item(self, item):
        self.backpack.append(item)
        print(f'You just got a {item.name}')
        return True

    # Remove some items from the player's backpack    
    def drop_item(self, item):
        self.backpack.remove(item)
        return True

    # Updating the striking power        
    def striking_power_check(self):
        a = 0
        for i in self.backpack:
            a += i.power
        self.item_striking_power = a
        b = self.item_striking_power + self.lucky_striking_power
        self.striking_power = self.natural_striking_power + b
        return self.striking_power

    # Updating the weight of the player's backpack        
    def weight_check(self):
        self.weight = 0
        weights = 0
        for i in self.backpack:
            weights += i.weight
        self.weight = weights
        return self.weight

    # Dropping some items from the player's backpack         
    def backpack_drop(self):
        drop_mode = 1
        while drop_mode == 1:
            clear_output()
            for i, item in enumerate(self.backpack):
                print("You can drop some items")
                print(f"{i} : {item.name}")
                time.sleep(0.5)
            drop_mode2 = 1
            while drop_mode2 == 1:
                try:
                    a = int(input("Please pick an item that you want to drop [Exit = -1]: "))
                    if a in range(len(self.backpack)) and a != -1:
                        item = self.backpack[a]
                        self.drop_item(item)
                        drop_mode2 = 0
                        time.sleep(0.5)
                    elif a == -1:
                        drop_mode2 = 0
                        drop_mode = 0
                    else: print("Please put the right integer number!")
                except ValueError:
                    print("Please put the right integer number!")

# NPC Class                         
class NPC:
    
    # Basic settings of a npc
    def __init__(self, name):
        self.name = name
        self.item_total_list = []

    # Append an item to the NPC's shop
    def add_item(self, item):
        self.item_total_list.append(item)

    # Players can sell their items       
    def selling(self, player):
        c = 1
        clear_output()
        while c == 1:
            print("")
            list_item = []
            for i in player.backpack:
                list_item.append(i.name)
            print("This is your backpack")
            time.sleep(1)
            for i, i2 in zip(list_item,range(len(list_item))):
                print(f"{i2}. {i}")
            try:
                time.sleep(1)
                a = int(input("Please pick the item which you want to sell from your backpack [exit = -1]: "))
                if a != -1 and a < len(list_item):
                    for item in self.item_total_list:
                        if player.backpack[a].name == item.name:
                            item.limit += 1
                            player.backpack.pop(a)
                            player.money += item.price
                            print(f"You sold {item.name} | Current balance: £{player.money}")
                            player.striking_power_check()
                            break
                elif a == -1:
                    c = 0
                elif a >= len(list_item):
                    print("Please put the integer number correctly!")
                else:
                    print("Please put the integer number!")
            except ValueError:
                print("Please put the right integer number!")
        # Print out the current backpack after they sell the items at the shop    
        list_backpack = []
        for i in player.backpack:
            list_backpack.append(i.name)
        clear_output()
        time.sleep(0.5)
        print(f"This is your backpack now: {list_backpack} and you have £{player.money}")

# General NPC Class (from NPC) 
class General_NPC(NPC):
    
    def __init__(self, name):
        super().__init__(name)
        self.item_total_list = []
        self.item_shop_list = []
        self.map_list = []
        self.room = f"{self.name}'s room"

    # Append an item to the NPC's shop2
    def add_shop(self, item):
        self.item_shop_list.append(item)

    # Append a map to the portal        
    def map_add(self, maps):
        self.map_list.append(maps)

    # Player can enter a map through this portal    
    def portal(self, player):
        player.striking_power_check()
        print(f"< Map list - Kingdom, Chosun | HP: {player.hp} | Striking power: {player.striking_power} >")
        time.sleep(0.5)
        for i, i2 in zip(self.map_list, range(len(self.map_list))):
            print(f"{i2}. {i.name} : Remain attempts: {i.attempts} | Reward: (£){i.reward} | Requirement: {i.requirement}")
        c = 1
        while c == 1:
            time.sleep(0.5)
            a = input("Select the map you want to try, stay here = -1 : ")
            try:
                if int(a) > -1 and int(a) < len(self.map_list) :
                    a = int(a)
                    if player.map_count >= self.map_list[a].map_number: # Check whether the player finished the privous map
                        if self.map_list[a].attempts > 0: # Check whether the player has enough attempts
                            player.current_location = 0
                            clear_output()
                            c = 0
                            self.map_list[a].start_game(player, self) # Start the map
                        else:
                            time.sleep(0.5)
                            print("You already used all your attempts, you cannot enter this map!")
                    else: print("You have to clear the previous stage :)")
                elif int(a) == -1:
                    c = 0
                else:
                    pass
            except ValueError:
                print("Please put the right integer number!")

    # Player can purchase an item through this shop        
    def shopping(self, player):
        print(f"Welcome to {self.name}'s Shop! you have £{player.money}")
        time.sleep(0.5)
        print("You can purchase some items below")
        time.sleep(0.5)
        for i, item in enumerate(self.item_shop_list):
            print(f"{i}: {item.name}")
            item.detail()
        c = 1
        while c == 1:
            a = int(input("Please pick an item that you want to buy, exit = -1 : "))
            time.sleep(0.5)
            try:
                if a != -1 and a < len(self.item_shop_list):
                    if player.money >= self.item_shop_list[a].price:
                        player.weight_check()
                        if type(self.item_shop_list[a]) == Portion:
                            if player.weight_limit >= player.weight + self.item_shop_list[a].weight: # Check the weight of the player's backpack
                                player.hp += self.item_shop_list[a].hp_up
                                player.money -= self.item_shop_list[a].price
                                print(f"You just bought {self.item_shop_list[a].name} | HP : {player.hp} | Balance: £{player.money}")
                                time.sleep(0.5)
                            else:
                                need_weight = (player.weight + self.item_shop_list[a].weight) - player.weight_limit
                                print(f"Your backpack is too heavy, you need to lighten it by {need_weight}kg")
                                time.sleep(0.5)                            
                        else:
                            if self.item_shop_list[a].limit > 0:
                                if player.weight_limit >= player.weight + self.item_shop_list[a].weight:
                                    self.item_shop_list[a].limit -= 1
                                    player.backpack.append(self.item_shop_list[a])
                                    player.money -= self.item_shop_list[a].price
                                    player.striking_power_check()
                                    print(f"You just bought {self.item_shop_list[a].name} | Striking Power: {player.striking_power} | Balance: £{player.money}")
                                    time.sleep(0.5)
                                else:
                                    need_weight = (player.weight + self.item_shop_list[a].weight) - player.weight_limit
                                    print(f"Your backpack is too heavy, you need to lighten it by {need_weight}kg")
                                    time.sleep(0.5)
                            else: 
                                print("You cannot buy this item anymore")
                                time.sleep(0.5)
                    else:
                        print("You don't have enough money!")
                elif a == -1:
                    clear_output()
                    print("Good bye!")
                    c = 0
                elif a > len(self.item_shop_list):
                    print("Please put the integer number correctly!")
                else:
                    print("Please put an integer number!")
            except ValueError:
                print("Please put the right integer number!")

    # Players can put their items to the locker                
    def deposit(self, player):
        list_item = []
        for i in player.backpack:
            list_item.append(i.name)
        print("This is your backpack")
        time.sleep(0.5)
        for i, i2 in zip(list_item,range(len(list_item))):
            print(f"{i2}. {i}")
        c = 1
        while c == 1:
            try:
                time.sleep(0.5)
                a = int(input("Please pick the item that you want to put in your locker [exit = -1]: "))
                time.sleep(0.5)
                if a != -1 and a < len(list_item):
                    b = player.backpack.pop(a)
                    player.locker_list.append(b)
                    player.striking_power_check()
                    c = 0
                elif a == -1:
                    c = 0
                elif a >= len(list_item):
                    print("Please put the integer number correctly!")
                else: print("Please put the integer number correctly!")
            except ValueError:
                print("Please put the right integer number!")
                
        list_locker = []
        for i in player.locker_list:
            list_locker.append(i.name)
        clear_output()
        print(f"This is your locker: {list_locker}")

    # Players can take out their items from the locker        
    def withdrawal(self, player):
        list_item = []
        for i in player.locker_list:
            list_item.append(i.name)
        print("This is your locker")
        time.sleep(0.5)
        for i, i2 in zip(list_item,range(len(list_item))):
            print(f"{i2}. {i}")
        c = 1
        while c == 1:
            try:
                time.sleep(0.5)
                a = int(input("Please pick the item that you want to put in your backpack [exit = -1]: "))
                if a != -1 and a < len(list_item):
                    if player.weight_limit >= player.weight + self.item_total_list[a].weight:
                        b = player.locker_list.pop(a)
                        player.backpack.append(b)
                        player.striking_power_check()
                        c = 0
                    else:
                        need_weight = (player.weight + self.item_total_list[a].weight) - player.weight_limit
                        time.sleep(0.5)
                        print(f"Your backpack is too heavy, you need to lighten it by {need_weight}kg")
                elif a == -1:
                    c = 0
                elif a >= len(list_item):
                    print("Please put the integer number correctly!")
                else:
                    print("Please put an integer number!")
            except ValueError:
                print("Please put the right integer number!")
                
        list_backpack = []
        for i in player.backpack:
            list_backpack.append(i.name)
        clear_output()
        time.sleep(0.5)
        print(f"This is your backpack now: {list_backpack}")

# Casino NPC Class (from NPC)
class Casino_NPC(NPC):
    
    def __init__(self, name):
        super().__init__(name)
        self.item_total_list = []
        self.item_shop_list = []
        self.map_list = []
        self.room = ""

    # Players can bet their money at this casino        
    def casino(self, player):
        casino_mode = 1
        clear_output()
        bet_rate = random.randint(2, 7)
        lucky_number = [1,2,3,4,5]
        time.sleep(0.5)
        print(f"£££ Welcome to Casino, I am today's dealer {self.name} £££")
        time.sleep(0.5)
        print(f"This time, you will earn {bet_rate} times the amount you wagered")
        while casino_mode == 1:
            try:
                time.sleep(0.5)
                bet_money = int(input(f"How much you want to bet?: [Leave: -1] | You have £{player.money} "))
                time.sleep(0.5)
                if bet_money > 0 and player.money > bet_money:
                    while True:
                        try:
                            player_pick = int(input("Please pick your lucky number [1, 2, 3, 4, 5] "))
                            if player_pick in lucky_number:
                                break
                            else: print("You can only pick a number between 1 to 5!")
                        except ValueError:
                            print("Please put the right integer number!")  
                    dealer_pick = random.randint(1, 5)
                    time.sleep(1.0)
                    if player_pick == dealer_pick:
                        player.money += bet_money*bet_rate
                        time.sleep(1.0)
                        print("£££££££££££££££££££££££££££££££££££££££££££££££££££££££")
                        print(f"Jackpot!, You just got £{bet_money*bet_rate} | Current money: £{player.money}")
                        print("£££££££££££££££££££££££££££££££££££££££££££££££££££££££")
                        time.sleep(2.0)
                    else:
                        player.money -= bet_money
                        print(f"HAHAHAHA, you lost £{bet_money}! | Current money: £{player.money}")
                elif bet_money > 0 and player.money < bet_money: # If the player doesn't have enough moeny, the player can sell their items at this casino
                    print("You don't have enough money!, You can sell your items or leave this casino ^_^")
                    time.sleep(0.5)
                    selling_or_leave = int(input("1. Selling | 2. Leave : "))
                    time.sleep(0.5)
                    if selling_or_leave == 1:
                        self.selling(player)
                    elif selling_or_leave ==2:
                        print("HAHAHAHA, Good bye!")
                        casino_mode = 0
                    else: print("Please put the right integer number!")
                elif bet_money == -1:
                    print("HAHAHAHA, Good bye!")
                    casino_mode = 0
                    break
                else: print("Please put the right integer number!")
            except ValueError:
                print("Please put the right integer number!")    


        