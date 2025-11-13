import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from datetime import datetime
from fpdf import FPDF
import base64
from io import BytesIO
import requests
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except:
    GRAPHVIZ_AVAILABLE = False

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
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.context_store = context_store
        self.chat = None
    
    def initialize_chat(self):
        """Initialize a new chat session"""
        self.chat = self.model.start_chat(history=[])
    
    def analyze_requirements(self, raw_input: str) -> Dict:
        """Analyze raw requirements input and extract structured requirements"""
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
            response_text = response.text.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            requirements = json.loads(response_text)
            req_id = f"REQ-{len(self.context_store.get_requirements()) + 1:03d}"
            self.context_store.add_requirement(req_id, requirements)
            
            return requirements
        except json.JSONDecodeError:
            return {"error": "Failed to parse requirements", "raw_response": response.text}
    
    def generate_user_stories(self, requirements: Optional[Dict] = None) -> List[Dict]:
        """Generate user stories from requirements using INVEST criteria"""
        if not self.chat:
            self.initialize_chat()
        
        if requirements is None:
            all_reqs = self.context_store.get_requirements()
            if not all_reqs:
                return {"error": "No requirements available"}
            requirements = all_reqs[-1]['data']
        
        prompt = f"""
        You are a Business Analyst Agent specializing in user story generation.
        
        Based on the following requirements, generate user stories following INVEST criteria:
        - Independent, Negotiable, Valuable, Estimatable, Small, Testable
        
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
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            user_stories_data = json.loads(response_text)
            
            for story in user_stories_data.get('user_stories', []):
                self.context_store.add_user_story(story['id'], story)
            
            return user_stories_data
        except json.JSONDecodeError:
            return {"error": "Failed to parse user stories", "raw_response": response.text}


class ArchitectAgent:
    """Agent for Design and Architecture"""
    
    def __init__(self, api_key: str, context_store: SharedContextStore):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.context_store = context_store
        self.chat = None
    
    def initialize_chat(self):
        """Initialize a new chat session"""
        self.chat = self.model.start_chat(history=[])
    
    def generate_architecture_design(self, requirements: Optional[Dict] = None) -> Dict:
        """Generate high-level architecture design from requirements"""
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
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            architecture = json.loads(response_text)
            self.context_store.add_design_artifact('architecture', architecture)
            
            return architecture
        except json.JSONDecodeError:
            return {"error": "Failed to parse architecture design", "raw_response": response.text}
    
    def generate_uml_class_diagram(self, architecture: Optional[Dict] = None) -> Dict:
        """Generate UML class diagram structure from architecture"""
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
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            uml_diagram = json.loads(response_text)
            self.context_store.add_design_artifact('uml_class_diagram', uml_diagram)
            
            return uml_diagram
        except json.JSONDecodeError:
            return {"error": "Failed to parse UML diagram", "raw_response": response.text}


class DocumentGenerator:
    """Generate PDF and TXT documents from requirements and user stories"""
    
    @staticmethod
    def generate_txt(requirements: Dict, user_stories: Dict, filename: str = "requirements_output.txt") -> str:
        """Generate TXT file"""
        output = []
        output.append("=" * 80)
        output.append("REQUIREMENTS ANALYSIS AND USER STORIES")
        output.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        output.append("\n")
        
        # Functional Requirements
        output.append("FUNCTIONAL REQUIREMENTS")
        output.append("-" * 80)
        for req in requirements.get('functional_requirements', []):
            output.append(f"\n{req['id']}: {req['description']}")
            output.append(f"Priority: {req['priority']}")
        
        # Non-Functional Requirements
        output.append("\n\nNON-FUNCTIONAL REQUIREMENTS")
        output.append("-" * 80)
        for req in requirements.get('non_functional_requirements', []):
            output.append(f"\n{req['id']} ({req['type']}): {req['description']}")
            output.append(f"Metric: {req.get('metric', 'N/A')}")
        
        # Constraints
        output.append("\n\nCONSTRAINTS")
        output.append("-" * 80)
        for constraint in requirements.get('constraints', []):
            output.append(f"- {constraint}")
        
        # User Stories
        output.append("\n\nUSER STORIES")
        output.append("-" * 80)
        for story in user_stories.get('user_stories', []):
            output.append(f"\n{story['id']} (Priority: {story['priority']}, Points: {story['story_points']})")
            output.append(f"As a {story['as_a']}")
            output.append(f"I want {story['i_want']}")
            output.append(f"So that {story['so_that']}")
            output.append("Acceptance Criteria:")
            for criteria in story['acceptance_criteria']:
                output.append(f"  - {criteria}")
            output.append(f"Linked Requirements: {', '.join(story['linked_requirements'])}")
        
        content = "\n".join(output)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename
    
    @staticmethod
    def generate_pdf(requirements: Dict, user_stories: Dict, filename: str = "requirements_output.pdf") -> str:
        """Generate PDF file"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Requirements Analysis and User Stories", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(10)
        
        # Functional Requirements
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Functional Requirements", ln=True)
        pdf.set_font("Arial", "", 10)
        
        for req in requirements.get('functional_requirements', []):
            pdf.set_font("Arial", "B", 10)
            pdf.multi_cell(0, 5, f"{req['id']} (Priority: {req['priority']})")
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, req['description'])
            pdf.ln(3)
        
        # Non-Functional Requirements
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Non-Functional Requirements", ln=True)
        pdf.set_font("Arial", "", 10)
        
        for req in requirements.get('non_functional_requirements', []):
            pdf.set_font("Arial", "B", 10)
            pdf.multi_cell(0, 5, f"{req['id']} - {req['type']}")
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, req['description'])
            pdf.multi_cell(0, 5, f"Metric: {req.get('metric', 'N/A')}")
            pdf.ln(3)
        
        # User Stories
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "User Stories", ln=True)
        
        for story in user_stories.get('user_stories', []):
            pdf.set_font("Arial", "B", 10)
            pdf.multi_cell(0, 5, f"{story['id']} (Priority: {story['priority']}, Points: {story['story_points']})")
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, f"As a {story['as_a']}")
            pdf.multi_cell(0, 5, f"I want {story['i_want']}")
            pdf.multi_cell(0, 5, f"So that {story['so_that']}")
            pdf.set_font("Arial", "B", 10)
            pdf.multi_cell(0, 5, "Acceptance Criteria:")
            pdf.set_font("Arial", "", 10)
            for criteria in story['acceptance_criteria']:
                pdf.multi_cell(0, 5, f"  - {criteria}")
            pdf.ln(5)
        
        pdf.output(filename)
        return filename


class DiagramGenerator:
    """Generate visual diagrams for architecture and UML"""
    
    @staticmethod
    def mermaid_to_image(mermaid_code: str) -> bytes:
        """Convert Mermaid code to PNG image using mermaid.ink API"""
        try:
            # Encode mermaid code to base64
            graphbytes = mermaid_code.encode("utf-8")
            base64_bytes = base64.b64encode(graphbytes)
            base64_string = base64_bytes.decode("ascii")
            
            # Use mermaid.ink API to generate image
            url = f"https://mermaid.ink/img/{base64_string}"
            
            # Fetch the image
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"Failed to generate image. Status code: {response.status_code}")
        except Exception as e:
            raise Exception(f"Error converting Mermaid to image: {str(e)}")
    
    @staticmethod
    def generate_architecture_diagram_mermaid(architecture: Dict) -> str:
        """Generate architecture diagram using Mermaid syntax"""
        mermaid_code = "graph TB\n"
        
        # Add components with proper escaped labels
        components = architecture.get('components', [])
        for i, component in enumerate(components):
            node_id = f"comp{i}"
            # Clean label: remove special characters, use only alphanumeric and spaces
            label = component['name'].replace('(', '').replace(')', '').replace('_', ' ')
            label = ''.join(c for c in label if c.isalnum() or c.isspace())
            mermaid_code += f"    {node_id}[\"{label}\"]\n"
        
        # Add relationships in a more logical pattern
        # Frontend apps connect to backend services
        if len(components) > 3:
            # Mobile/Web apps
            for i in range(min(4, len(components))):
                if i < len(components) - 4:
                    mermaid_code += f"    comp{i} --> comp{min(4, len(components)-1)}\n"
            
            # Backend services interconnections
            for i in range(4, len(components) - 1):
                mermaid_code += f"    comp{i} -.-> comp{i+1}\n"
        else:
            # Simple linear flow for smaller architectures
            for i in range(len(components) - 1):
                mermaid_code += f"    comp{i} --> comp{i+1}\n"
        
        # Add storage
        storage = architecture.get('data_storage', {})
        if storage.get('databases'):
            dbs = storage.get('databases', [])
            db_label = dbs[0] if dbs else 'Database'
            mermaid_code += f"    db[(\"{db_label}\")]\n"
            # Connect backend services to database
            if len(components) > 4:
                for i in range(4, min(len(components), 8)):
                    mermaid_code += f"    comp{i} --> db\n"
            else:
                mermaid_code += f"    comp{len(components)-1} --> db\n"
        
        # Add styling at the end
        mermaid_code += "\n    classDef frontend fill:#e3f2fd,stroke:#1976d2,stroke-width:2px\n"
        mermaid_code += "    classDef backend fill:#fff3e0,stroke:#f57c00,stroke-width:2px\n"
        mermaid_code += "    classDef database fill:#e8f5e9,stroke:#388e3c,stroke-width:2px\n"
        
        # Apply styles
        for i in range(min(4, len(components))):
            mermaid_code += f"    class comp{i} frontend\n"
        for i in range(4, len(components)):
            mermaid_code += f"    class comp{i} backend\n"
        if storage.get('databases'):
            mermaid_code += "    class db database\n"
        
        return mermaid_code
    
    @staticmethod
    def generate_uml_diagram_mermaid(uml_data: Dict) -> str:
        """Generate UML class diagram using Mermaid syntax"""
        mermaid_code = "classDiagram\n"
        
        # Add classes with proper escaping
        for cls in uml_data.get('classes', []):
            class_name = cls['name'].replace(' ', '').replace('-', '')
            # Ensure class name is valid (alphanumeric only)
            class_name = ''.join(c for c in class_name if c.isalnum())
            
            mermaid_code += f"    class {class_name} {{\n"
            
            # Add attributes (limit to 5)
            for attr in cls.get('attributes', [])[:5]:
                visibility = '+' if attr.get('visibility') == 'public' else '-'
                attr_type = attr.get('type', 'String').replace(' ', '')
                attr_name = attr.get('name', 'attribute').replace(' ', '')
                mermaid_code += f"        {visibility}{attr_type} {attr_name}\n"
            
            # Add methods (limit to 5)
            for method in cls.get('methods', [])[:5]:
                visibility = '+' if method.get('visibility') == 'public' else '-'
                method_name = method.get('name', 'method').replace(' ', '')
                return_type = method.get('return_type', 'void').replace(' ', '')
                mermaid_code += f"        {visibility}{method_name}() {return_type}\n"
            
            mermaid_code += "    }\n"
        
        # Add relationships with proper class names
        for rel in uml_data.get('relationships', []):
            from_cls = rel['from'].replace(' ', '').replace('-', '')
            to_cls = rel['to'].replace(' ', '').replace('-', '')
            from_cls = ''.join(c for c in from_cls if c.isalnum())
            to_cls = ''.join(c for c in to_cls if c.isalnum())
            
            rel_type = rel.get('type', 'association')
            
            if rel_type == 'inheritance':
                mermaid_code += f"    {to_cls} <|-- {from_cls}\n"
            elif rel_type == 'composition':
                mermaid_code += f"    {from_cls} *-- {to_cls}\n"
            elif rel_type == 'aggregation':
                mermaid_code += f"    {from_cls} o-- {to_cls}\n"
            else:
                mermaid_code += f"    {from_cls} --> {to_cls}\n"
        
        return mermaid_code
    
    @staticmethod
    def generate_architecture_diagram(architecture: Dict) -> bytes:
        """Generate architecture diagram using Graphviz (fallback)"""
        if not GRAPHVIZ_AVAILABLE:
            raise Exception("Graphviz not available. Please install it or use Mermaid diagrams.")
        
        dot = graphviz.Digraph(comment='System Architecture')
        dot.attr(rankdir='TB', size='10,10')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        # Add components
        for component in architecture.get('components', []):
            label = f"{component['name']}\n{component['responsibility'][:50]}..."
            dot.node(component['name'], label)
        
        # Add relationships based on communication patterns
        components = [c['name'] for c in architecture.get('components', [])]
        for i in range(len(components) - 1):
            dot.edge(components[i], components[i + 1])
        
        # Render to PNG
        return dot.pipe(format='png')
    
    @staticmethod
    def generate_uml_class_diagram(uml_data: Dict) -> bytes:
        """Generate UML class diagram using Graphviz (fallback)"""
        if not GRAPHVIZ_AVAILABLE:
            raise Exception("Graphviz not available. Please install it or use Mermaid diagrams.")
        
        dot = graphviz.Digraph(comment='UML Class Diagram')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='record')
        
        # Add classes
        for cls in uml_data.get('classes', []):
            # Build class label
            attrs = [f"- {attr['name']}: {attr['type']}" for attr in cls.get('attributes', [])]
            methods = [f"+ {method['name']}(): {method['return_type']}" for method in cls.get('methods', [])]
            
            label = f"{{{cls['name']}|"
            if attrs:
                label += "|".join(attrs)
            label += "|"
            if methods:
                label += "|".join(methods)
            label += "}"
            
            dot.node(cls['name'], label)
        
        # Add relationships
        for rel in uml_data.get('relationships', []):
            rel_type = rel['type']
            if rel_type == 'inheritance':
                dot.edge(rel['from'], rel['to'], arrowhead='empty')
            elif rel_type == 'composition':
                dot.edge(rel['from'], rel['to'], arrowhead='diamond')
            elif rel_type == 'aggregation':
                dot.edge(rel['from'], rel['to'], arrowhead='odiamond')
            else:
                dot.edge(rel['from'], rel['to'])
        
        # Render to PNG
        return dot.pipe(format='png')


def main():
    st.set_page_config(page_title="Multi-Agent SDLC System", layout="wide", page_icon="🤖")
    
    st.title("🤖 Multi-Agent SDLC System")
    st.markdown("### AI-Powered Requirements Analysis & Architecture Design")
    
    # Initialize session state
    if 'context_store' not in st.session_state:
        st.session_state.context_store = SharedContextStore()
    if 'requirements' not in st.session_state:
        st.session_state.requirements = None
    if 'user_stories' not in st.session_state:
        st.session_state.user_stories = None
    if 'architecture' not in st.session_state:
        st.session_state.architecture = None
    if 'uml_diagram' not in st.session_state:
        st.session_state.uml_diagram = None
    
    # Get API Key from environment
    api_key = os.getenv('API_KEY_CHATBOT', '')
    
    if not api_key:
        st.error("⚠️ API_KEY_CHATBOT not found in environment variables. Please add it to your .env file.")
        st.code("API_KEY_CHATBOT=your_api_key_here", language="bash")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    phase = st.sidebar.radio("Select Phase", ["Phase 1: Requirements Analysis", "Phase 2: Architecture Design"])
    
    if phase == "Phase 1: Requirements Analysis":
        st.header("📋 Phase 1: Requirements Analysis")
        
        # Input area
        st.subheader("Input Your Requirements")
        raw_input = st.text_area(
            "Enter your project requirements:",
            height=200,
            placeholder="Example: We need to build an e-commerce platform for selling books online...",
            help="Describe your project requirements in natural language"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_btn = st.button("🔍 Analyze Requirements", type="primary")
        
        if analyze_btn and raw_input:
            with st.spinner("🤖 Business Analyst Agent is analyzing requirements..."):
                ba_agent = BusinessAnalystAgent(api_key, st.session_state.context_store)
                st.session_state.requirements = ba_agent.analyze_requirements(raw_input)
            
            with st.spinner("📝 Generating user stories..."):
                st.session_state.user_stories = ba_agent.generate_user_stories()
            
            st.success("✅ Analysis complete!")
        
        # Display results
        if st.session_state.requirements and st.session_state.user_stories:
            st.subheader("📊 Analysis Results")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Functional Requirements", "Non-Functional Requirements", "User Stories", "Download"])
            
            with tab1:
                st.markdown("### Functional Requirements")
                for req in st.session_state.requirements.get('functional_requirements', []):
                    with st.expander(f"**{req['id']}** - Priority: {req['priority']}"):
                        st.write(req['description'])
            
            with tab2:
                st.markdown("### Non-Functional Requirements")
                for req in st.session_state.requirements.get('non_functional_requirements', []):
                    with st.expander(f"**{req['id']}** - {req['type'].upper()}"):
                        st.write(req['description'])
                        st.caption(f"Metric: {req.get('metric', 'N/A')}")
            
            with tab3:
                st.markdown("### User Stories")
                for story in st.session_state.user_stories.get('user_stories', []):
                    with st.expander(f"**{story['id']}** - {story['priority']} Priority ({story['story_points']} points)"):
                        st.markdown(f"**As a** {story['as_a']}")
                        st.markdown(f"**I want** {story['i_want']}")
                        st.markdown(f"**So that** {story['so_that']}")
                        st.markdown("**Acceptance Criteria:**")
                        for criteria in story['acceptance_criteria']:
                            st.markdown(f"- {criteria}")
                        st.caption(f"Linked to: {', '.join(story['linked_requirements'])}")
            
            with tab4:
                st.markdown("### 📥 Download Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 Generate TXT Report"):
                        txt_file = DocumentGenerator.generate_txt(
                            st.session_state.requirements,
                            st.session_state.user_stories
                        )
                        with open(txt_file, 'rb') as f:
                            st.download_button(
                                label="Download TXT",
                                data=f,
                                file_name=txt_file,
                                mime="text/plain"
                            )
                
                with col2:
                    if st.button("📑 Generate PDF Report"):
                        pdf_file = DocumentGenerator.generate_pdf(
                            st.session_state.requirements,
                            st.session_state.user_stories
                        )
                        with open(pdf_file, 'rb') as f:
                            st.download_button(
                                label="Download PDF",
                                data=f,
                                file_name=pdf_file,
                                mime="application/pdf"
                            )
    
    else:  # Phase 2
        st.header("🏗️ Phase 2: Architecture Design")
        
        if not st.session_state.requirements:
            st.warning("⚠️ Please complete Phase 1 first!")
            return
        
        col1, col2 = st.columns([1, 5])
        with col1:
            design_btn = st.button("🎨 Generate Architecture", type="primary")
        
        if design_btn:
            with st.spinner("🤖 Architect Agent is designing the system..."):
                architect_agent = ArchitectAgent(api_key, st.session_state.context_store)
                st.session_state.architecture = architect_agent.generate_architecture_design()
            
            with st.spinner("📐 Generating UML class diagram..."):
                st.session_state.uml_diagram = architect_agent.generate_uml_class_diagram()
            
            st.success("✅ Architecture design complete!")
        
        # Display architecture
        if st.session_state.architecture and st.session_state.uml_diagram:
            tab1, tab2, tab3 = st.tabs(["Architecture Overview", "Visual Diagrams", "JSON Data"])
            
            with tab1:
                arch = st.session_state.architecture
                
                st.markdown(f"### Architecture Type: **{arch.get('architecture_type', 'N/A').upper()}**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🧩 Components")
                    for comp in arch.get('components', []):
                        with st.expander(f"**{comp['name']}**"):
                            st.write(f"**Technology:** {comp['technology']}")
                            st.write(f"**Responsibility:** {comp['responsibility']}")
                            st.write(f"**Interfaces:** {', '.join(comp.get('interfaces', []))}")
                
                with col2:
                    st.markdown("#### 💾 Data Storage")
                    storage = arch.get('data_storage', {})
                    st.write(f"**Databases:** {', '.join(storage.get('databases', []))}")
                    st.write(f"**Rationale:** {storage.get('rationale', 'N/A')}")
                    
                    st.markdown("#### 🔄 Communication Patterns")
                    for pattern in arch.get('communication_patterns', []):
                        st.write(f"- {pattern}")
            
            with tab2:
                st.markdown("### 📊 Architecture Diagram")
                
                try:
                    mermaid_arch = DiagramGenerator.generate_architecture_diagram_mermaid(st.session_state.architecture)
                    
                    # Display Mermaid code
                    with st.expander("📝 View Mermaid Code", expanded=False):
                        st.code(mermaid_arch, language="mermaid")
                        st.download_button(
                            label="📋 Download Mermaid Code (.mmd)",
                            data=mermaid_arch,
                            file_name="architecture_diagram.mmd",
                            mime="text/plain",
                            key="arch_mermaid_code"
                        )
                    
                    # Generate and display PNG image
                    st.info("🎨 Generating diagram image...")
                    with st.spinner("Converting Mermaid to PNG..."):
                        try:
                            arch_img = DiagramGenerator.mermaid_to_image(mermaid_arch)
                            st.image(arch_img, caption="System Architecture Diagram", use_container_width=True)
                            
                            # Download button for PNG
                            st.download_button(
                                label="💾 Download Architecture Diagram (PNG)",
                                data=arch_img,
                                file_name="architecture_diagram.png",
                                mime="image/png",
                                key="arch_png"
                            )
                            st.success("✅ Diagram generated successfully!")
                        except Exception as img_error:
                            st.warning(f"⚠️ Could not generate PNG image: {img_error}")
                            st.info("💡 You can still copy the Mermaid code above and visualize it at: https://mermaid.live/")
                    
                except Exception as e:
                    st.error(f"Error generating architecture diagram: {e}")
                
                # Try Graphviz if available
                # if GRAPHVIZ_AVAILABLE:
                #     st.markdown("---")
                #     st.markdown("### 🎨 Graphviz Version (Alternative)")
                #     try:
                #         arch_img_gv = DiagramGenerator.generate_architecture_diagram(st.session_state.architecture)
                #         st.image(arch_img_gv, caption="System Architecture Diagram (Graphviz)")
                #         st.download_button(
                #             label="💾 Download Graphviz PNG",
                #             data=arch_img_gv,
                #             file_name="architecture_diagram_graphviz.png",
                #             mime="image/png",
                #             key="arch_graphviz"
                #         )
                #     except Exception as e:
                #         st.caption(f"Graphviz unavailable: {e}")
                
                st.markdown("---")
                st.markdown("### 🎯 UML Class Diagram")
                
                try:
                    mermaid_uml = DiagramGenerator.generate_uml_diagram_mermaid(st.session_state.uml_diagram)
                    
                    # Display Mermaid code
                    with st.expander("📝 View Mermaid Code", expanded=False):
                        st.code(mermaid_uml, language="mermaid")
                        st.download_button(
                            label="📋 Download Mermaid Code (.mmd)",
                            data=mermaid_uml,
                            file_name="uml_class_diagram.mmd",
                            mime="text/plain",
                            key="uml_mermaid_code"
                        )
                    
                    # Generate and display PNG image
                    st.info("🎨 Generating UML diagram image...")
                    with st.spinner("Converting Mermaid to PNG..."):
                        try:
                            uml_img = DiagramGenerator.mermaid_to_image(mermaid_uml)
                            st.image(uml_img, caption="UML Class Diagram", use_container_width=True)
                            
                            # Download button for PNG
                            st.download_button(
                                label="💾 Download UML Diagram (PNG)",
                                data=uml_img,
                                file_name="uml_class_diagram.png",
                                mime="image/png",
                                key="uml_png"
                            )
                            st.success("✅ UML diagram generated successfully!")
                        except Exception as img_error:
                            st.warning(f"⚠️ Could not generate PNG image: {img_error}")
                            st.info("💡 You can still copy the Mermaid code above and visualize it at: https://mermaid.live/")
                    
                except Exception as e:
                    st.error(f"Error generating UML diagram: {e}")
                
                # Try Graphviz if available
                # if GRAPHVIZ_AVAILABLE:
                #     st.markdown("---")
                #     st.markdown("### 🎨 Graphviz Version (Alternative)")
                #     try:
                #         uml_img_gv = DiagramGenerator.generate_uml_class_diagram(st.session_state.uml_diagram)
                #         st.image(uml_img_gv, caption="UML Class Diagram (Graphviz)")
                #         st.download_button(
                #             label="💾 Download Graphviz PNG",
                #             data=uml_img_gv,
                #             file_name="uml_class_diagram_graphviz.png",
                #             mime="image/png",
                #             key="uml_graphviz"
                #         )
                #     except Exception as e:
                #         st.caption(f"Graphviz unavailable: {e}")
            
            with tab3:
                st.markdown("### Architecture JSON")
                st.json(st.session_state.architecture)
                
                st.markdown("### UML Diagram JSON")
                st.json(st.session_state.uml_diagram)


if __name__ == "__main__":
    main()