use std::io::Write;
use chrono::Local;
use maud::{html, Markup, PreEscaped};
use plotly::Plot;

/// Struct to represent a section of the report
pub struct ReportSection {
    title: String,
    content_blocks: Vec<Markup>, // Multiple content blocks (text or plots)
}

impl ReportSection {
    /// Create a new section with a title
    pub fn new(title: &str) -> Self {
        ReportSection {
            title: title.to_string(),
            content_blocks: Vec::new(),
        }
    }

    /// Add a block of content (text, HTML, etc.)
    pub fn add_content(&mut self, content: Markup) {
        self.content_blocks.push(content);
    }

    /// Add a plot to the section
    pub fn add_plot(&mut self, plot: Plot) {
        self.content_blocks.push(html! {
            div style="width: 800px; height: 500px;" {
                (PreEscaped(plot.to_inline_html(None)))
            }
        });
    }

    /// Render the section as HTML
    fn render(&self) -> Markup {
        html! {
            div {
                h2 { (self.title) }
                @for block in &self.content_blocks {
                    (block)
                }
            }
        }
    }
}



/// Struct to represent the entire report
pub struct Report {
    software_name: String,
    version: String,
    software_logo: Option<String>,
    title: String,
    sections: Vec<ReportSection>,
}

impl Report {
    /// Create a new report with a title
    pub fn new(software_name: &str, version: &str, software_logo: Option<&str>, title: &str) -> Self {
        Report {
            software_name: software_name.to_string(),
            version: version.to_string(),
            software_logo: software_logo.map(|s| s.to_string()),
            title: title.to_string(),
            sections: Vec::new(),
        }
    }

    /// Add a section to the report
    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    /// Render the entire report as HTML
    fn render(&self) -> Markup {
        let current_date = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    
        html! {
            html {
                head {
                    title { (self.title) }
                    script src="https://cdn.plot.ly/plotly-latest.min.js" {}
    
                    // Banner CSS
                    style {
                        (PreEscaped("
                            body {
                                font-family: Arial, sans-serif;
                            }
                            .banner {
                                display: flex;
                                align-items: center;
                                justify-content: space-between;
                                padding: 15px;
                                background: linear-gradient(135deg, #4a90e2, #145da0);
                                border-radius: 12px;
                                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                                color: white;
                                margin-bottom: 20px;
                                max-width: 100%;
                                overflow: hidden;
                            }
                            .banner img {
                                max-height: 150px; /* Size of logo */
                                width: auto;
                                height: auto;
                                margin-right: 15px;
                            }
                            .banner-text {
                                flex-grow: 1;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;
                            }
                            .banner-text h2 {
                                font-size: 50px;
                                margin: 0;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;
                            }
                            .banner-text p {
                                font-size: 24px;
                                margin: 0;
                                opacity: 0.8;
                            }
                        "))
                    }
                    
                    // Tabs CSS
                    style {
                        (PreEscaped("
                            body {
                                font-family: Arial, sans-serif;
                            }
                            .tabs {
                                display: flex;
                                border-bottom: 2px solid #ddd;
                            }
                            .tab {
                                padding: 10px 20px;
                                cursor: pointer;
                                border: none;
                                background: none;
                                font-size: 16px;
                                font-weight: bold;
                                color: #444;
                                transition: 0.3s;
                            }
                            .tab:hover {
                                color: #000;
                            }
                            .tab.active {
                                border-bottom: 3px solid #007bff;
                                color: #007bff;
                            }
                            .tab-content {
                                display: none;
                                padding: 20px;
                                border-radius: 10px;
                            }
                            .tab-content.active {
                                display: block;
                            }
                        "))
                    }
    
                    // JavaScript for tabs
                    script {
                        (PreEscaped("
                            function showTab(tabId) {
                                let tabs = document.querySelectorAll('.tab-content');
                                tabs.forEach(tab => tab.classList.remove('active'));
    
                                let buttons = document.querySelectorAll('.tab');
                                buttons.forEach(btn => btn.classList.remove('active'));
    
                                document.getElementById(tabId).classList.add('active');
                                document.querySelector(`[data-tab='${tabId}']`).classList.add('active');
                            }
                        "))
                    }
                }
    
                body {
                    // Banner Section
                    div class="banner" {
                        @if let Some(ref logo) = self.software_logo {
                            img src=(logo) alt="Software Logo";
                        }
                        div class="banner-text" {
                            h2 { (self.software_name) " v" (self.version) }
                            p class="timestamp" { "Generated on: " (current_date) }
                        }
                    }                    
    
                    // Tabs Navigation
                    div class="tabs" {
                        @for (i, section) in self.sections.iter().enumerate() {
                            button class="tab" data-tab=(format!("tab{}", i)) onclick=(format!("showTab('tab{}')", i)) {
                                (section.title.clone())
                            }
                        }
                    }
    
                    // Tab Content
                    @for (i, section) in self.sections.iter().enumerate() {
                        div id=(format!("tab{}", i)) class={@if i == 0 { "tab-content active" } @else { "tab-content" }} {
                            (section.render())
                        }
                    }
                    
                }
            }
        }
    }
    

    /// Save the report to an HTML file
    pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(filename)?;
        file.write_all(self.render().into_string().as_bytes())?;
        Ok(())
    }
}