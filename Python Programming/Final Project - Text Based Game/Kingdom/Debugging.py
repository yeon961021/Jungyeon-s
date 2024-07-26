#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 9 20:07:33 2023

@author: ijeong-yeon
"""
# uniitest library for DDebugging
import unittest

from Kingdom import *

class Test(unittest.TestCase):
    
    # Game setting for Debugging
    def setUp(self):
        self.map1 = Normal_map("Main Entrance of Chosun Palace", 5, 3, 1, 10, 1) # map
        self.map2 = Traning_map("Training Room 1", 30, 5, 10, 50) # Training map
        self.map2.map_setting(map_list = ["1F Corridor", "2F Corridor", "Chichester1 016", "Chichester1 017", "Chichester1 021", "Chichester1 023"],
                     monster_list = [("Undergraduate Student", 2),("Postgraduate Student",5),("Exchange Student",7),
                        ("TA",12),("Professor", 14)])
        self.player1 = Player("Jungyeon", 27)
        self.player2 = Player("Omer", 31)
        self.item1 = Weapon("Revolver", 200, 10, 1, 300)
        self.item2 = Key("Training Key1", 1, 0.5, 1, 10)
        self.map2.key_setting(self.item2)
        
    # Debugging some parts of this game
    def test(self):
        self.assertTrue(self.map1.hp_check(self.player1))
        self.player1.hp = 0
        self.assertFalse(self.map1.hp_check(self.player1))
        
        self.player2.add_item(self.item2)
        self.assertTrue(self.map2.check_key(self.player2))
        
        self.map2.win_count = 6
        self.assertTrue(self.map2.win_check(self.player2))
        
        self.assertTrue(self.player2.add_item(self.item1))
        self.assertEqual(self.player2.striking_power_check(), 300)
        self.assertTrue(self.player2.weight_check(), 10)
        self.assertTrue(self.player2.drop_item(self.item1))
        self.assertEqual(self.player2.striking_power_check(), 0)
        self.assertTrue(self.player2.weight_check(), 0)                   
    
def main():
    a = Test()
    a.setUp()
    a.test()
    
main()