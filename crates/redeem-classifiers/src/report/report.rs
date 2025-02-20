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
            (maud::DOCTYPE)
            html {
                head {
                    title { (self.title) }
                    script src="https://cdn.plot.ly/plotly-latest.min.js" {}
                    script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.4/jquery.min.js" {}
                    script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js" {}
                    link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css" {}
                    script src="https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js" {}
    
                    // JavaScript for DataTables and CSV export
                    script {
                        r#"
                        $(document).ready(function() {
                            let table = $('#dataTable').DataTable({
                                paging: true,
                                searching: true,
                                ordering: true
                            });
                            
                            $('#downloadCsv').on('click', function() {
                                let csv = [];
                                let headers = [];
                                $('#dataTable thead th').each(function() {
                                    headers.push($(this).text());
                                });
                                csv.push(headers.join(','));
                                
                                $('#dataTable tbody tr').each(function() {
                                    let row = [];
                                    $(this).find('td').each(function() {
                                        row.push('"' + $(this).text() + '"');
                                    });
                                    csv.push(row.join(','));
                                });
                                
                                let csvContent = csv.join('\n');
                                let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                                saveAs(blob, 'table_data.csv');
                            });
                        });
                        "#
                    }
    
                    // JavaScript for tabs
                    script {
                        (PreEscaped(r#"
                            function showTab(tabId) {
                                document.querySelectorAll('.tab-content').forEach(function(tab) {
                                    tab.classList.remove('active');
                                });
                    
                                document.querySelectorAll('.tab').forEach(function(tab) {
                                    tab.classList.remove('active');
                                });
                    
                                document.getElementById(tabId).classList.add('active');
                                document.querySelector(`[data-tab='${tabId}']`).classList.add('active');
                            }
                        "#))
                    }
                    

                    // CSS styles
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
                                max-height: 100px;
                                width: auto;
                                height: auto;
                                margin-right: 15px;
                            }
                            .banner-text h2 {
                                font-size: 36px;
                                margin: 0;
                                white-space: nowrap;
                            }
                            .banner-text p {
                                font-size: 16px;
                                margin: 0;
                                opacity: 0.8;
                            }
                            .tabs {
                                display: flex;
                                border-bottom: 2px solid #ddd;
                            }
                            .tab {
                                padding: 10px 20px;
                                cursor: pointer;
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
                            }
                            .tab-content.active {
                                display: block;
                            }
                        "))
                    }
                }
                
                body {
                    div class="banner" {
                        @if let Some(ref logo) = self.software_logo {
                            img src=(logo) alt="Software Logo";
                        }
                        div class="banner-text" {
                            h2 { (self.software_name) " v" (self.version) }
                            p class="timestamp" { "Generated on: " (current_date) }
                        }
                    }
                    
                    div class="tabs" {
                        @for (i, section) in self.sections.iter().enumerate() {
                            button class="tab" data-tab=(format!("tab{}", i)) onclick=(format!("showTab('tab{}')", i)) {
                                (section.title.clone())
                            }
                        }
                    }
    
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