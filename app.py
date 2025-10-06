import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class SharedContextStore:
    """Shared memory for agents to communicate and maintain traceability"""
    def __init__(self):
        self.context = {
            'requirements': [],
            'user_stories': [],
            'design_artifacts': {},
            'traceability': {}
        }
    
    def add_requirement(self, req_id: str, requirement: Dict):
        """Add a requirement to the shared context"""
        self.context['requirements'].append({
            'id': req_id,
            'data': requirement
        })
    
    def add_user_story(self, story_id: str, story: Dict):
        """Add a user story to the shared context"""
        self.context['user_stories'].append({
            'id': story_id,
            'data': story
        })
    
    def add_design_artifact(self, artifact_type: str, artifact: Dict):
        """Add a design artifact to the shared context"""
        if artifact_type not in self.context['design_artifacts']:
            self.context['design_artifacts'][artifact_type] = []
        self.context['design_artifacts'][artifact_type].append(artifact)
    
    def get_requirements(self) -> List[Dict]:
        """Retrieve all requirements"""
        return self.context['requirements']
    
    def get_user_stories(self) -> List[Dict]:
        """Retrieve all user stories"""
        return self.context['user_stories']
    
    def get_design_artifacts(self, artifact_type: Optional[str] = None) -> Dict:
        """Retrieve design artifacts"""
        if artifact_type:
            return self.context['design_artifacts'].get(artifact_type, [])
        return self.context['design_artifacts']


class BusinessAnalystAgent:
    """Agent for Requirements Elicitation and Analysis"""
    
    def __init__(self, api_key: str, context_store: SharedContextStore):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.context_store = context_store
        self.chat = None
    
    def initialize_chat(self):
        """Initialize a new chat session"""
        self.chat = self.model.start_chat(history=[])
    
    def analyze_requirements(self, raw_input: str) -> Dict:
        """
        Analyze raw requirements input and extract structured requirements
        
        Args:
            raw_input: Raw requirements text from stakeholders
            
        Returns:
            Dictionary containing structured requirements
        """
        if not self.chat:
            self.initialize_chat()
        
        prompt = f"""
        You are a Business Analyst Agent specializing in requirements elicitation and analysis.
        
        Analyze the following raw requirements and extract:
        1. Functional Requirements (with unique IDs)
        2. Non-Functional Requirements (performance, security, scalability)
        3. Constraints and Dependencies
        4. Stakeholder Concerns
        
        Raw Input:
        {raw_input}
        
        Provide output in JSON format with the following structure:
        {{
            "functional_requirements": [
                {{"id": "FR-001", "description": "...", "priority": "High/Medium/Low"}}
            ],
            "non_functional_requirements": [
                {{"id": "NFR-001", "type": "performance/security/etc", "description": "...", "metric": "..."}}
            ],
            "constraints": ["..."],
            "dependencies": ["..."],
            "stakeholder_concerns": ["..."]
        }}
        """
        
        response = self.chat.send_message(prompt)
        
        try:
            # Parse the response and clean it
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            requirements = json.loads(response_text)
            
            # Store in context store
            req_id = f"REQ-{len(self.context_store.get_requirements()) + 1:03d}"
            self.context_store.add_requirement(req_id, requirements)
            
            return requirements
        except json.JSONDecodeError:
            return {"error": "Failed to parse requirements", "raw_response": response.text}
    
    def generate_user_stories(self, requirements: Optional[Dict] = None) -> List[Dict]:
        """
        Generate user stories from requirements using INVEST criteria
        
        Args:
            requirements: Optional specific requirements, otherwise uses all from context
            
        Returns:
            List of user stories
        """
        if not self.chat:
            self.initialize_chat()
        
        if requirements is None:
            all_reqs = self.context_store.get_requirements()
            if not all_reqs:
                return {"error": "No requirements available"}
            requirements = all_reqs[-1]['data']  # Use most recent
        
        prompt = f"""
        You are a Business Analyst Agent specializing in user story generation.
        
        Based on the following requirements, generate user stories following INVEST criteria:
        - Independent
        - Negotiable
        - Valuable
        - Estimatable
        - Small
        - Testable
        
        Requirements:
        {json.dumps(requirements, indent=2)}
        
        Generate user stories in the format:
        {{
            "user_stories": [
                {{
                    "id": "US-001",
                    "as_a": "type of user",
                    "i_want": "goal",
                    "so_that": "benefit",
                    "acceptance_criteria": ["criterion 1", "criterion 2"],
                    "priority": "High/Medium/Low",
                    "story_points": 1-13,
                    "linked_requirements": ["FR-001", "NFR-001"]
                }}
            ]
        }}
        """
        
        response = self.chat.send_message(prompt)
        
        try:
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            user_stories_data = json.loads(response_text)
            
            # Store in context store
            for story in user_stories_data.get('user_stories', []):
                self.context_store.add_user_story(story['id'], story)
            
            return user_stories_data
        except json.JSONDecodeError:
            return {"error": "Failed to parse user stories", "raw_response": response.text}


class ArchitectAgent:
    """Agent for Design and Architecture"""
    
    def __init__(self, api_key: str, context_store: SharedContextStore):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.context_store = context_store
        self.chat = None
    
    def initialize_chat(self):
        """Initialize a new chat session"""
        self.chat = self.model.start_chat(history=[])
    
    def generate_architecture_design(self, requirements: Optional[Dict] = None) -> Dict:
        """
        Generate high-level architecture design from requirements
        
        Args:
            requirements: Optional specific requirements, otherwise uses all from context
            
        Returns:
            Architecture design document
        """
        if not self.chat:
            self.initialize_chat()
        
        if requirements is None:
            all_reqs = self.context_store.get_requirements()
            if not all_reqs:
                return {"error": "No requirements available"}
            requirements = all_reqs[-1]['data']
        
        prompt = f"""
        You are an Architect Agent specializing in software architecture design.
        
        Based on the following requirements, design a high-level architecture including:
        1. System Components and their responsibilities
        2. Component Interactions and Communication Patterns
        3. Data Flow and Storage Strategy
        4. Technology Stack Recommendations
        5. Design Patterns to be used
        6. Scalability and Performance Considerations
        
        Requirements:
        {json.dumps(requirements, indent=2)}
        
        Provide output in JSON format:
        {{
            "architecture_type": "microservices/monolithic/layered/etc",
            "components": [
                {{
                    "name": "ComponentName",
                    "responsibility": "...",
                    "technology": "...",
                    "interfaces": ["..."]
                }}
            ],
            "communication_patterns": ["REST API", "Message Queue", "etc"],
            "data_storage": {{
                "databases": ["PostgreSQL", "Redis", "etc"],
                "rationale": "..."
            }},
            "technology_stack": {{
                "backend": ["..."],
                "frontend": ["..."],
                "infrastructure": ["..."]
            }},
            "design_patterns": ["Singleton", "Factory", "Observer", "etc"],
            "scalability_strategy": "...",
            "performance_considerations": ["..."]
        }}
        """
        
        response = self.chat.send_message(prompt)
        
        try:
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            architecture = json.loads(response_text)
            
            # Store in context store
            self.context_store.add_design_artifact('architecture', architecture)
            
            return architecture
        except json.JSONDecodeError:
            return {"error": "Failed to parse architecture design", "raw_response": response.text}
    
    def generate_uml_class_diagram(self, architecture: Optional[Dict] = None) -> Dict:
        """
        Generate UML class diagram structure from architecture
        
        Args:
            architecture: Optional specific architecture, otherwise uses latest from context
            
        Returns:
            UML class diagram structure
        """
        if not self.chat:
            self.initialize_chat()
        
        if architecture is None:
            arch_artifacts = self.context_store.get_design_artifacts('architecture')
            if not arch_artifacts:
                return {"error": "No architecture design available"}
            architecture = arch_artifacts[-1]
        
        prompt = f"""
        You are an Architect Agent specializing in UML diagram generation.
        
        Based on the following architecture design, create a UML class diagram structure:
        
        Architecture:
        {json.dumps(architecture, indent=2)}
        
        Generate class definitions with:
        - Class names
        - Attributes (with types and visibility)
        - Methods (with parameters, return types, and visibility)
        - Relationships (inheritance, composition, aggregation, association)
        
        Provide output in JSON format:
        {{
            "classes": [
                {{
                    "name": "ClassName",
                    "attributes": [
                        {{"name": "attributeName", "type": "string", "visibility": "private/public/protected"}}
                    ],
                    "methods": [
                        {{
                            "name": "methodName",
                            "parameters": [{{"name": "param", "type": "type"}}],
                            "return_type": "type",
                            "visibility": "public"
                        }}
                    ]
                }}
            ],
            "relationships": [
                {{
                    "from": "ClassA",
                    "to": "ClassB",
                    "type": "inheritance/composition/aggregation/association",
                    "cardinality": "1..1/1..*/*..*/etc"
                }}
            ]
        }}
        """
        
        response = self.chat.send_message(prompt)
        
        try:
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            uml_diagram = json.loads(response_text)
            
            # Store in context store
            self.context_store.add_design_artifact('uml_class_diagram', uml_diagram)
            
            return uml_diagram
        except json.JSONDecodeError:
            return {"error": "Failed to parse UML diagram", "raw_response": response.text}
    
    def verify_design(self, requirements: Optional[Dict] = None) -> Dict:
        """
        Verify design against requirements for traceability and consistency
        
        Args:
            requirements: Optional specific requirements to verify against
            
        Returns:
            Verification report
        """
        if not self.chat:
            self.initialize_chat()
        
        if requirements is None:
            all_reqs = self.context_store.get_requirements()
            if not all_reqs:
                return {"error": "No requirements available"}
            requirements = all_reqs[-1]['data']
        
        architecture = self.context_store.get_design_artifacts('architecture')
        if not architecture:
            return {"error": "No architecture design to verify"}
        
        prompt = f"""
        You are a Design Verifier Agent.
        
        Verify that the architecture design addresses all requirements:
        
        Requirements:
        {json.dumps(requirements, indent=2)}
        
        Architecture:
        {json.dumps(architecture[-1], indent=2)}
        
        Check for:
        1. Requirement Coverage - Are all functional requirements addressed?
        2. Non-functional Requirement Satisfaction
        3. Design Consistency
        4. Missing Components
        5. Potential Issues or Risks
        
        Provide verification report in JSON:
        {{
            "requirement_coverage": {{
                "covered": ["FR-001", "FR-002"],
                "missing": ["FR-003"],
                "coverage_percentage": 85
            }},
            "nfr_satisfaction": {{
                "satisfied": [{{"id": "NFR-001", "rationale": "..."}}],
                "not_satisfied": [{{"id": "NFR-002", "reason": "..."}}]
            }},
            "consistency_check": {{
                "is_consistent": true,
                "issues": []
            }},
            "risks": ["risk 1", "risk 2"],
            "recommendations": ["recommendation 1"]
        }}
        """
        
        response = self.chat.send_message(prompt)
        
        try:
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            verification = json.loads(response_text)
            
            return verification
        except json.JSONDecodeError:
            return {"error": "Failed to parse verification report", "raw_response": response.text}


# Example usage
def main():
    # Initialize shared context store
    context_store = SharedContextStore()
    
    # Get API key from environment
    api_key = os.getenv('API_KEY_CHATBOT')
    
    if not api_key:
        print("Error: API_KEY_CHATBOT not found in environment variables")
        return
    
    # Initialize agents
    ba_agent = BusinessAnalystAgent(api_key, context_store)
    architect_agent = ArchitectAgent(api_key, context_store)
    
    # Example: Analyze requirements
    print("=== Phase 1: Requirements Analysis ===\n")
    
    raw_requirements = """
    We need to build an e-commerce platform for selling books online.
    Users should be able to browse books by category, search for specific titles,
    add books to cart, and checkout securely. The system should handle 10000 concurrent users.
    Payment processing must be secure and support multiple payment methods.
    The platform should be responsive and work on mobile devices.
    """
    
    print("Analyzing requirements...")
    requirements = ba_agent.analyze_requirements(raw_requirements)
    print(json.dumps(requirements, indent=2))
    
    print("\n=== Generating User Stories ===\n")
    user_stories = ba_agent.generate_user_stories()
    print(json.dumps(user_stories, indent=2))
    
    # Example: Generate architecture
    print("\n=== Phase 2: Architecture Design ===\n")
    
    print("Generating architecture design...")
    architecture = architect_agent.generate_architecture_design()
    print(json.dumps(architecture, indent=2))
    
    print("\n=== Generating UML Class Diagram ===\n")
    uml = architect_agent.generate_uml_class_diagram()
    print(json.dumps(uml, indent=2))
    
    print("\n=== Verifying Design ===\n")
    verification = architect_agent.verify_design()
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()