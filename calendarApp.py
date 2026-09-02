import os
import sys
import time
import json
import sqlite3
import subprocess
import threading
from datetime import datetime
from PIL import ImageGrab
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from pynput import keyboard
import customtkinter as ctk

class InitialAssignmentEvent(BaseModel):
    summary: str = Field(description="Name of the assignment, test, or schedule event.")
    date: str = Field(description="The event date formatted strictly as YYYY-MM-DD.")
    start_time: str = Field(description="The event start time formatted strictly as HH:MM. Default to 23:59 if unstated.")
    end_time: str = Field(description="The event end time formatted strictly as HH:MM. Set to 'None' if unstated or unknown.")
    description: str = Field(description="Notes and related info")
    search_keywords: List[str] = Field(description="2 or 3 distinct keywords to locate matching text updates.")

class InitialScheduleResponse(BaseModel):
    events: List[InitialAssignmentEvent] = Field(description="A list of detected events on the screen.")

class FinalAuditedResponse(BaseModel):
    has_changes: bool = Field(description="True if recent communications contradict or modify the initial scheduling data.")
    audit_source: str = Field(description="Who said it? Set to 'None' if no adjustments.")
    audit_context: str = Field(description="Short summary of what context was found. Set to 'None' if no adjustments.")
    final_summary: str = Field(description="The finalized title of the event.")
    final_date: str = Field(description="The finalized date formatted strictly as YYYY-MM-DD.")
    final_start_time: str = Field(description="The finalized start time formatted strictly as HH:MM.")
    final_end_time: str = Field(description="The finalized end time formatted strictly as HH:MM. Set to 'None' if unstated.")
    final_description: str = Field(description="The comprehensive description.")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

keyFilePath = os.path.expanduser("~/.gemini_key")

class CalendarAgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Calendar Agent")
        self.geometry("500x680")
        self.resizable(False, False)

        self.currentHotkey = "`" 
        self.hotkeyListener = None
        self.geminiClient = None

        self.setup_ui()
        self.load_saved_api_key()
        self.start_hotkey_listener()

    def setup_ui(self):
        self.titleLabel = ctk.CTkLabel(
            self, text="AI Calendar Auditor", font=ctk.CTkFont(size=22, weight="bold")
        )
        self.titleLabel.pack(pady=(25, 5))

        self.subtitleLabel = ctk.CTkLabel(
            self, text="Press your assigned hotkey anywhere to audit screen deadlines", font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.subtitleLabel.pack(pady=(0, 20))

        self.apiFrame = ctk.CTkFrame(self, corner_radius=10)
        self.apiFrame.pack(fill="x", padx=30, pady=(0, 10))

        self.apiLabel = ctk.CTkLabel(self.apiFrame, text="Gemini API Key:", font=ctk.CTkFont(size=12, weight="bold"))
        self.apiLabel.pack(anchor="w", padx=15, pady=(10, 2))

        self.apiEntry = ctk.CTkEntry(self.apiFrame, placeholder_text="AIzaSy...", show="•")
        self.apiEntry.pack(fill="x", padx=15, pady=(0, 5))
        
        self.saveKeyBtn = ctk.CTkButton(self.apiFrame, text="Save & Update Key", command=self.save_api_key, height=28)
        self.saveKeyBtn.pack(fill="x", padx=15, pady=(5, 10))

        self.hotkeyFrame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.hotkeyFrame.pack(fill="x", padx=30, pady=(5, 10))

        self.hotkeyLabel = ctk.CTkLabel(self.hotkeyFrame, text="Trigger Hotkey:", font=ctk.CTkFont(size=13, weight="bold"))
        self.hotkeyLabel.pack(side="left", padx=(5, 10))

        self.hotkeyBtn = ctk.CTkButton(
            self.hotkeyFrame, 
            text=f"[ {self.currentHotkey} ]", 
            state="disabled",
            width=200,
            fg_color="#3A3A3C",
            text_color_disabled="white"
        )
        self.hotkeyBtn.pack(side="right")

        self.statusFrame = ctk.CTkFrame(self, corner_radius=10)
        self.statusFrame.pack(fill="x", padx=30, pady=5)

        self.statusDot = ctk.CTkLabel(
            self.statusFrame, text="●", text_color="#EF4444", font=ctk.CTkFont(size=18)
        )
        self.statusDot.pack(side="left", padx=(15, 5), pady=12)

        self.statusText = ctk.CTkLabel(
            self.statusFrame, text="Missing API Key Activation", font=ctk.CTkFont(size=14, weight="normal")
        )
        self.statusText.pack(side="left", pady=12)

        self.scanButton = ctk.CTkButton(
            self, text="Scan Screen Now", command=self.trigger_manual_scan, font=ctk.CTkFont(weight="bold"), height=40, state="disabled"
        )
        self.scanButton.pack(fill="x", padx=30, pady=(15, 10))

        self.logLabel = ctk.CTkLabel(self, text="Activity Log", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.logLabel.pack(anchor="w", padx=35, pady=(10, 5))

        self.logTextbox = ctk.CTkTextbox(self, height=180, corner_radius=8, font=ctk.CTkFont(family="Courier", size=12))
        self.logTextbox.pack(fill="both", padx=30, pady=(0, 25))
        self.logTextbox.configure(state="disabled")

    def load_saved_api_key(self):
        if os.path.exists(keyFilePath):
            try:
                with open(keyFilePath, "r") as f:
                    savedKey = f.read().strip()
                if savedKey:
                    self.apiEntry.insert(0, savedKey)
                    self.initialize_gemini_client(savedKey)
            except Exception as e:
                self.log(f"Failed loading cached key config: {e}")

    def save_api_key(self):
        inputKey = self.apiEntry.get().strip()
        if not inputKey:
            self.log("Error: API Key field cannot be blank.")
            return
        try:
            with open(keyFilePath, "w") as f:
                f.write(inputKey)
            self.initialize_gemini_client(inputKey)
            self.log("API Key stored securely and initialized successfully.")
        except Exception as e:
            self.log(f"Failed to write local key file config: {e}")

    def initialize_gemini_client(self, apiKey: str):
        try:
            self.geminiClient = genai.Client(api_key=apiKey)
            self.scanButton.configure(state="normal")
            self.update_status("Agent Background Listener Active", "#10B981")
        except Exception as e:
            self.log(f"SDK Client initialization failed: {e}")

    def start_hotkey_listener(self):
        try:
            self.hotkeyListener = keyboard.GlobalHotKeys({self.currentHotkey: self.trigger_manual_scan})
            self.hotkeyListener.start()
        except Exception as e:
            self.log(f"Failed to bind hotkey: {e}. Resetting to fallback [ ` ]")
            self.currentHotkey = "`"
            self.hotkeyListener = keyboard.GlobalHotKeys({self.currentHotkey: self.trigger_manual_scan})
            self.hotkeyListener.start()

    def trigger_manual_scan(self):
        if self.geminiClient:
            threading.Thread(target=self.process_screenshot_pipeline, daemon=True).start()

    def log(self, logMessage: str):
        self.logTextbox.configure(state="normal")
        self.logTextbox.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {logMessage}\n")
        self.logTextbox.see("end")
        self.logTextbox.configure(state="disabled")

    def update_status(self, displayText: str, statusColor: str):
        self.statusDot.configure(text_color=statusColor)
        self.statusText.configure(text=displayText)

    def process_screenshot_pipeline(self):
        self.scanButton.configure(state="disabled")
        self.update_status("Capturing screen & scanning...", "#3B82F6")
        
        try:
            self.log("Capturing Screen...")
            screenImg = ImageGrab.grab()
            currentYear = datetime.now().year
            
            self.log("Parsing screen context via Gemini...")
            promptStage1 = f"Extract visible academic deadlines. Provide search keywords. Current Year: {currentYear}."
            
            geminiResponse = self.geminiClient.models.generate_content(
                model='gemini-2.5-flash',
                contents=[screenImg, promptStage1],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InitialScheduleResponse,
                ),
            )
            initialData = json.loads(geminiResponse.text)
            detectedEvents = initialData.get("events", [])
            
            if not detectedEvents:
                self.log("No events detected on screen.")
                self.reset_ui_status()
                return

            self.log(f"Found {len(detectedEvents)} prospective events. Verifying variations...")

            for singleEvent in detectedEvents:
                self.log(f"Auditing details for: '{singleEvent['summary']}'")
                searchKeywords = singleEvent.get('search_keywords', [singleEvent['summary'].split()[0]])
                localContext = self.get_recent_imessages_local(searchKeywords)
                
                promptStage2 = f"""
                Verify this calendar data from a screenshot: {json.dumps(singleEvent)}
                Against these recent messages: {localContext}
                Identify if details have changed. Synthesize into schema format.
                """
                
                auditedResp = self.geminiClient.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=promptStage2,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=FinalAuditedResponse,
                        temperature=0.1
                    ),
                )
                auditedData = json.loads(auditedResp.text)

                if auditedData['has_changes']:
                    self.log(f"Conflict found for '{singleEvent['summary']}'! Prompting user...")
                    userChoice = self.request_user_approval_local(
                        title=singleEvent['summary'],
                        source=auditedData['audit_source'],
                        context=auditedData['audit_context'],
                        oldDate=singleEvent['date'],
                        newDate=auditedData['final_date']
                    )
                    
                    if userChoice == "Approve":
                        self.log("User approved adjustments. Updating Apple Calendar...")
                        self.delete_apple_calendar_event_local(singleEvent['summary'], singleEvent['date'])
                    else:
                        self.log("User rejected changes. Saving initial screen capture parameters.")
                        auditedData.update({
                            'final_summary': singleEvent['summary'],
                            'final_date': singleEvent['date'],
                            'final_start_time': singleEvent['start_time'],
                            'final_end_time': singleEvent['end_time'],
                            'final_description': singleEvent['description']
                        })
                
                isSuccess = self.create_apple_calendar_event_local(auditedData)
                if isSuccess:
                    self.log(f"Success: Saved '{auditedData['final_summary']}' to Calendar.")
                else:
                    self.log(f"Error: Could not write event '{auditedData['final_summary']}'")
                
                time.sleep(0.3)
                
        except Exception as e:
            self.log(f"Pipeline Exception: {str(e)}")
        
        self.reset_ui_status()

    def reset_ui_status(self):
        self.scanButton.configure(state="normal")
        self.update_status("Agent Background Listener Active", "#10B981")

    def get_recent_imessages_local(self, searchKeywords: List[str]) -> str:
        chatDbPath = os.path.expanduser("~/Library/Messages/chat.db")
        if not os.path.exists(chatDbPath):
            return "Local database file missing."
        compiledTexts = []
        queryParts = " OR ".join(["text LIKE ?" for _ in searchKeywords])
        sqlCommand = f"""
            SELECT text, datetime(date/1000000000 + 978307200, 'unixepoch', 'localtime') as msg_date
            FROM message WHERE ({queryParts}) AND text IS NOT NULL ORDER BY date DESC LIMIT 10
        """
        queryParams = [f"%{word}%" for word in searchKeywords]
        try:
            dbConn = sqlite3.connect(chatDbPath)
            dbCursor = dbConn.cursor()
            dbCursor.execute(sqlCommand, queryParams)
            for dbRow in dbCursor.fetchall():
                compiledTexts.append(f"[{dbRow[1]}] iMessage: {dbRow[0]}")
            dbConn.close()
        except Exception as e:
            return f"Database error: {e}"
        return "\n".join(compiledTexts) if compiledTexts else "No updates found."

    def request_user_approval_local(self, title, source, context, oldDate, newDate) -> str:
        dialogMsg = f"Conflict discovered for '{title}'!\n\n• Source: {source}\n• Context: \"{context}\"\n\nChange date from [{oldDate}] to [{newDate}]?"
        cleanMsg = dialogMsg.replace('"', '\\"')
        appleScript = f'''
        tell application "System Events"
            activate
            set promptWindow to display dialog "{cleanMsg}" with title "Calendar Agent Verification" buttons {{"Deny", "Approve"}} default button "Approve" with icon caution
            return button returned of promptWindow
        end tell
        '''
        scriptResult = subprocess.run(["osascript", "-e", appleScript], capture_output=True, text=True)
        return scriptResult.stdout.strip()

    def delete_apple_calendar_event_local(self, eventSummary, dateStr) -> bool:
        try:
            dateObj = datetime.strptime(dateStr, "%Y-%m-%d")
            yearVal, monthVal, dayVal = dateObj.year, dateObj.month, dateObj.day
        except Exception: 
            return False
        appleScript = f'''
        tell application "Calendar"
            set targetCalendar to calendar "Test"
            set targetDate to (current date)
            set year of targetDate to {yearVal}\nset month of targetDate to {monthVal}\nset day of targetDate to {dayVal}
            set hours of targetDate to 0\nset minutes of targetDate to 0\nset seconds of targetDate to 0
            tell targetCalendar
                delete (every event whose (summary contains "{eventSummary}") and (start date ≥ targetDate) and (start date < targetDate + 1 * days))
            end tell
        end tell
        '''
        return subprocess.run(["osascript", "-e", appleScript], capture_output=True).returncode == 0

    def create_apple_calendar_event_local(self, eventData) -> bool:
        eventSummary = eventData['final_summary'].replace('"', '\\"')
        eventDescription = eventData['final_description'].replace('"', '\\"')
        try:
            startDateObj = datetime.strptime(f"{eventData['final_date']} {eventData['final_start_time']}", "%Y-%m-%d %H:%M")
            yearVal, monthVal, dayVal, hourVal, minVal = startDateObj.year, startDateObj.month, startDateObj.day, startDateObj.hour, startDateObj.minute
            
            endTimeRaw = eventData.get('final_end_time')
            if endTimeRaw and endTimeRaw != 'None':
                endDateObj = datetime.strptime(f"{eventData['final_date']} {endTimeRaw}", "%Y-%m-%d %H:%M")
                endYear, endMonth, endDay, endHour, endMin = endDateObj.year, endDateObj.month, endDateObj.day, endDateObj.hour, endDateObj.minute
                
                endScriptPayload = f'''
                set eventEnd to (current date)
                set year of eventEnd to {endYear}
                set month of eventEnd to {endMonth}
                set day of eventEnd to {endDay}
                set hours of eventEnd to {endHour}
                set minutes of eventEnd to {endMin}
                set seconds of eventEnd to 0
                '''
            else:
                endScriptPayload = "set eventEnd to eventStart + (30 * minutes)"
                
        except Exception: 
            return False
            
        appleScript = f'''
        tell application "Calendar"
            set targetCalendar to calendar "Test"
            set eventStart to (current date)
            set year of eventStart to {yearVal}\nset month of eventStart to {monthVal}\nset day of eventStart to {dayVal}
            set hours of eventStart to {hourVal}\nset minutes of eventStart to {minVal}\nset seconds of eventStart to 0
            {endScriptPayload}
            tell targetCalendar
                make new event with properties {{summary:"{eventSummary}", start date:eventStart, end date:eventEnd, description:"{eventDescription}"}}
            end tell
            save
        end tell
        '''
        return subprocess.run(["osascript", "-e", appleScript], capture_output=True).returncode == 0

if __name__ == "__main__":
    appInstance = CalendarAgentApp()
    try:
        appInstance.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)
