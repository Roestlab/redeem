use chrono::Local;
use maud::{html, Markup, PreEscaped};
use plotly::Plot;
use rand::{distributions::Alphanumeric, Rng};
use std::io::Write;

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
        let plot_id: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(10)
            .map(char::from)
            .collect();

        self.content_blocks.push(html! {
            div class="plot-wrapper" {
                div id=(plot_id.clone()) class="plot-container" {
                    (PreEscaped(plot.to_inline_html(Some(&plot_id))))
                }
            }
            script {
                (PreEscaped(format!(r#"
                    function resizePlot() {{
                        let plotDiv = document.getElementById('{plot_id}');
                        if (plotDiv) {{
                            let width = window.innerWidth * 0.8;
                            Plotly.relayout(plotDiv, {{ width: width }});
                        }}
                    }}
                    window.addEventListener('resize', resizePlot);
                    resizePlot(); // Call initially
                "#)))
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
    pub fn new(
        software_name: &str,
        version: &str,
        software_logo: Option<&str>,
        title: &str,
    ) -> Self {
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
                    script src="https://cdn.datatables.net/colresize/1.0.0/dataTables.colResize.min.js" {}
                    link rel="stylesheet" href="https://cdn.datatables.net/colResize/1.0.0/css/colResize.dataTables.min.css" {}
                    script src="https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js" {}

                    // JavaScript for DataTables and CSV export
                    script {
                        (PreEscaped(r#"
                            $(document).ready(function() {
                                let table = $('#dataTable').DataTable({
                                    paging: true,
                                    searching: true,
                                    ordering: true,
                                    scrollX: true,
                                    autoWidth: false,  // Ensures DataTables doesn't override widths
                                    colResize: {
                                        enable: true,  // Enable column resizing
                                        resizeTable: true
                                    }
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
                        "#))
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
                    // CSS for the table container
                    style {
                        (PreEscaped("
                            .table-container {
                                width: 100%;
                                overflow-x: auto; /* Enable horizontal scrolling */
                                white-space: nowrap; /* Prevent line breaks in cells */
                                border: 1px solid #ddd; /* Optional: Add a border */
                                padding: 10px;
                            }
                            table {
                                width: 100%;
                                border-collapse: collapse;
                            }
                            table.display {
                                width: 100% 
                                table-layout: fixed;
                                border-collapse: collapse;
                            }

                            .dataTables_scrollHeadInner {
                                width: 100% !important;
                            }
                        "))
                    }

                    // CSS for the plot container
                    style {
                        (PreEscaped("
                            .plot-wrapper {
                                width: 100%;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                position: relative;
                            }

                            .plot-container {
                                width: 100%;
                                // max-width: 1200px; /* Prevents it from getting too large */
                                height: 600px; /* Adjust as needed */
                                position: relative;
                                overflow: hidden; /* Prevents content from spilling */
                                // border: 1px solid #ccc; /* Optional: Helps visualize layout */
                            }
                        "))
                    }

                    // CSS for the report
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

#[cfg(test)]

mod tests {
    use super::*;
    use crate::report::plots::plot_scatter;

    #[test]
    fn test_report() {
        let mut report = Report::new("Redeem", "1.0", Some("logo.png"), "My Report");

        let mut section1 = ReportSection::new("Section 1");
        section1.add_content(html! {
            p { "This is the first section of the report." }
        });

        // create table
        let table = html! {
            table class="display" id="dataTable" {
                thead {
                    tr {
                        th { "Name" }
                        th { "Age" }
                        th { "City" }
                        th { "Country" }
                        th { "Occupation" }
                        th { "Salary" }
                        th { "Join Date" }
                        th { "Active" }
                        th { "Actions" }
                        th { "Actions" }
                        th { "Actions" }
                    }
                }
                tbody {
                    tr {
                        td { "JohnMichaelbrunovalentinemark Beckham" }
                        td { "30" }
                        td { "New York" }
                        td { "USA" }
                        td { "Engineer" }
                        td { "100,000" }
                        td { "2022-01-01" }
                        td { "Yes" }
                        td { "Edit | Delete" }
                        td { "Edit | Delete" }
                        td { "Edit | Delete" }
                    }
                    tr {
                        td { "Jane Smith" }
                        td { "25" }
                        td { "Los Angeles" }
                        td { "USA" }
                        td { "Designer" }
                        td { "80,000" }
                        td { "2022-02-15" }
                        td { "No" }
                        td { "Edit | Delete" }
                        td { "Edit | Delete" }
                        td { "Edit | Delete" }
                    }
                }
            }
        };
        section1.add_content(table.clone());

        report.add_section(section1);

        // Add a scatter plot
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 7.0, 3.0, 9.0, 10.0],
            vec![1.0, 12.0, 13.0, 14.0, 15.0],
        ];
        let y = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![11.0, 12.0, 13.0, 14.0, 15.0],
        ];
        let labels = vec![
            "file1".to_string(),
            "file2".to_string(),
            "file3".to_string(),
        ];
        let title = "Scatter Plot";
        let x_title = "X";
        let y_title = "Y";

        let plot = plot_scatter(&x, &y, labels, title, x_title, y_title).unwrap();

        let mut section2 = ReportSection::new("Section 2");
        section2.add_plot(plot.clone());

        // add some content latin
        section2.add_content(html! {
            p { "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ac nisl..." }
        });

        section2.add_content(table);

        // add another plot (the same one)
        section2.add_plot(plot);

        report.add_section(section2);

        report.save_to_file("report.html").unwrap();
    }
}
